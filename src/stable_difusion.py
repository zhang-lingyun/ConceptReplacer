from transformers import CLIPTextModel, CLIPTokenizer, logging

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import LMSDiscreteScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy
import random

class StableDiffusion(nn.Module):
    def __init__(
        self,
        sd_version="2.0",
        step_guidance=None,
        attention_layers_to_use=[],
    ):
        super().__init__()

        self.sd_version = sd_version
        print(f"[INFO] loading stable diffusion...")

        if self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "1.4":
            print(f"[INFO] loading stable diffusion v1.4...")
            model_key = "CompVis/stable-diffusion-v1-4"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder"
        )
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        del self.vae.decoder
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        self.scheduler.set_timesteps(100)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.020)
        self.max_step = int(self.num_train_timesteps * 0.980)
        if step_guidance is not None:
            self.min_step, self.max_step = step_guidance

        self.alphas = self.scheduler.alphas_cumprod  # for convenience
        self.device = None
        self.device1 = None
        print(f"[INFO] loaded stable diffusion!")

        self.noise = None

        self.attention_maps = {}

        def create_nested_hook_for_attention_modules(n):
            def hook(module, input, output):
                self.attention_maps[n] = output[1]

            return hook

        self.handles = []
        for module in attention_layers_to_use:
            self.handles.append(
                eval("self.unet." + module).register_forward_hook(
                    create_nested_hook_for_attention_modules(module)
                )
            )

    def change_hooks(self, attention_layers_to_use):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.attention_maps = {}

        def create_nested_hook_for_attention_modules_with_detach(n):
            def hook(module, input, output):
                self.attention_maps[n] = output[1].detach()

            return hook

        for module in attention_layers_to_use:
            self.handles.append(
                eval("self.unet." + module).register_forward_hook(
                    create_nested_hook_for_attention_modules_with_detach(module)
                )
            )

    def setup(self, device, device1=None):
        self.device1 = device if device1 is None else device1
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(self.device1)
        self.alphas = self.alphas.to(device)
        self.device = device

    def get_text_embeds(self, prompt, negative_prompt, **kwargs):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.set_grad_enabled(False):
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.set_grad_enabled(False):
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]

        return uncond_embeddings, text_embeddings

    def get_attention_map(
        self,
        raw_attention_maps,
        output_size=256,
        token_ids=(2,),
        average_layers=True,
        train=True,
        apply_softmax=True,
        cross_atten="merge",
        beta=None,
    ):
        cross_attention_maps = {}
        self_attention_maps = {}
        resnets = {}
        for layer in raw_attention_maps:
            if layer.endswith("attn2"):  # cross attentions
                split_attention_maps = torch.stack(
                    raw_attention_maps[layer].chunk(2), dim=0
                )
                _, channel, img_embed_len, text_embed_len = split_attention_maps.shape
                if train:
                    if apply_softmax:
                        reshaped_split_attention_maps = (
                            split_attention_maps[:, :, :, torch.tensor(list(token_ids))]
                            .softmax(dim=-1)
                            .reshape(
                                2,  # because of chunk
                                channel,
                                int(math.sqrt(img_embed_len)),
                                int(math.sqrt(img_embed_len)),
                                len(token_ids),
                            )
                            .permute(0, 1, 4, 2, 3)
                        )
                    else:
                        reshaped_split_attention_maps = (
                            split_attention_maps[:, :, :, torch.tensor(list(token_ids))]
                            .reshape(
                                2,  # because of chunk
                                channel,
                                int(math.sqrt(img_embed_len)),
                                int(math.sqrt(img_embed_len)),
                                len(token_ids),
                            )
                            .permute(0, 1, 4, 2, 3)
                        )
                else:
                    reshaped_split_attention_maps = (
                        split_attention_maps.softmax(dim=-1)[
                            :, :, :, torch.tensor(list(token_ids))
                        ]
                        .reshape(
                            2,  # because of chunk
                            channel,
                            int(math.sqrt(img_embed_len)),
                            int(math.sqrt(img_embed_len)),
                            len(token_ids),
                        )
                        .permute(0, 1, 4, 2, 3)
                    )

                resized_reshaped_split_attention_maps_0 = (
                    torch.nn.functional.interpolate(
                        reshaped_split_attention_maps[0],
                        size=output_size,
                        mode="bilinear",
                        antialias=True,
                    ).mean(dim=0)
                )
                resized_reshaped_split_attention_maps_1 = (
                    torch.nn.functional.interpolate(
                        reshaped_split_attention_maps[1],
                        size=output_size,
                        mode="bilinear",
                        antialias=True,
                    ).mean(dim=0)
                )
                resized_reshaped_split_attention_maps = torch.stack(
                    [
                        resized_reshaped_split_attention_maps_0,
                        resized_reshaped_split_attention_maps_1,
                    ],
                    dim=0,
                )
                cross_attention_maps[layer] = resized_reshaped_split_attention_maps

            elif layer.endswith("attn1"):  # self attentions
                channel, img_embed_len, img_embed_len = raw_attention_maps[layer].shape
                split_attention_maps = raw_attention_maps[layer][channel // 2 :]
                reshaped_split_attention_maps = (
                    split_attention_maps.softmax(dim=-1)
                    .reshape(
                        # 2,  # because of chunk
                        channel // 2,
                        int(math.sqrt(img_embed_len)),
                        int(math.sqrt(img_embed_len)),
                        img_embed_len,
                    )
                    .permute(0, 3, 1, 2)
                )
                resized_reshaped_split_attention_maps = torch.nn.functional.interpolate(
                    reshaped_split_attention_maps, size=output_size, mode="bilinear", antialias=True,
                )
                resized_reshaped_split_attention_maps = (
                    resized_reshaped_split_attention_maps.mean(dim=0)
                )
                self_attention_maps[layer] = resized_reshaped_split_attention_maps
            elif layer.endswith("conv1") or layer.endswith("conv2"):  # resnets
                resnets[layer] = raw_attention_maps[layer].detach()
        sd_cross_attention_maps = [None, None]
        sd_self_attention_maps = None

        if len(cross_attention_maps.values()) > 0:
            if average_layers:
                if cross_atten=="low":
                    sd_cross_attention_maps = (
                        torch.stack(list(cross_attention_maps.values())[:5], dim=0)
                        .mean(dim=0)
                        .to(self.device)
                    )
                elif cross_atten=="merge":
                    sd_cross_attention_maps_low, sd_cross_attention_maps_high = list(cross_attention_maps.values())[:5],list(cross_attention_maps.values())[5:]

                    sd_cross_attention_maps_low = (
                        torch.stack(sd_cross_attention_maps_low, dim=0)
                        .mean(dim=0)
                        .detach()
                        .to(self.device)
                    )
                    sd_cross_attention_maps_high = (
                        torch.stack(sd_cross_attention_maps_high, dim=0)
                        .mean(dim=0)
                        .to(self.device)
                    ).sigmoid()
                    
                    sd_cross_attention_maps = (1 - beta) * sd_cross_attention_maps_low + beta * (sd_cross_attention_maps_low * sd_cross_attention_maps_high)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if len(self_attention_maps.values()) > 0:
            sd_self_attention_maps = torch.stack(
                list(self_attention_maps.values()), dim=0
            ).mean(dim=0)

        if len(resnets.values()) > 0:
            r = list(map(lambda x: x.to(self.device), list(resnets.values())))

        return (
            sd_cross_attention_maps[0],
            sd_cross_attention_maps[1],
            sd_self_attention_maps,
        )

    def train_step(
        self,
        text_embeddings,
        input_image,
        guidance_scale=100,
        t=None,
        generate_new_noise=True,
        attention_map=None,
        latents=None,
        attention_output_size=256,
        token_ids=(2,),
        average_layers=True,
        train=True,
        apply_softmax=True,
        cross_atten="merge",
        beta=None,
    ):
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')
        if not (input_image.shape[-2] == 512 and input_image.shape[-1] == 512):
            input_image = F.interpolate(
                input_image, (512, 512), mode="bilinear", align_corners=False, antialias=True
            )
        
        times = [999.0000, 988.9091, 978.8182, 968.7273, 958.6364, 948.5455, 938.4545,
            928.3636, 918.2727, 908.1818, 898.0909, 888.0000, 877.9091, 867.8182,
            857.7273, 847.6364, 837.5455, 827.4545, 817.3636, 807.2727, 797.1818,
            787.0909, 777.0000, 766.9091, 756.8182, 746.7273, 736.6364, 726.5455,
            716.4545, 706.3636, 696.2727, 686.1818, 676.0909, 666.0000, 655.9091,
            645.8182, 635.7273, 625.6364, 615.5455, 605.4545, 595.3636, 585.2727,
            575.1818, 565.0909, 555.0000, 544.9091, 534.8182, 524.7273, 514.6364,
            504.5455, 494.4545, 484.3636, 474.2727, 464.1818, 454.0909, 444.0000,
            433.9091, 423.8182, 413.7273, 403.6364, 393.5455, 383.4545, 373.3636,
            363.2727, 353.1818, 343.0909, 333.0000, 322.9091, 312.8182, 302.7273,
            292.6364, 282.5455, 272.4545, 262.3636, 252.2727, 242.1818, 232.0909,
            222.0000, 211.9091, 201.8182, 191.7273, 181.6364, 171.5455, 161.4545,
            151.3636, 141.2727, 131.1818, 121.0909, 111.0000, 100.9091,  90.8182,
            80.7273,  70.6364,  60.5455,  50.4545,  40.3636,  30.2727,  20.1818,
            10.0909,  0.0000]
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if len(t) == 2:
            while True:
                chosed_t = random.choice(times)
                if t[0] <= chosed_t <= t[1]:
                    chosed_t = torch.tensor([chosed_t]).to(torch.float64)
                    break
        else:
            chosed_t = t
            chosed_t = chosed_t.to(torch.float64)

        chosed_t = chosed_t.to(self.device)

        # encode image into latents with vae, requires grad!
        if latents is None:
            latents = self.encode_imgs(input_image)
        if attention_map is not None:
            latents = latents * attention_map.to(self.device)

        with torch.set_grad_enabled(True):
            # add noise
            if generate_new_noise:
                noise = torch.randn_like(latents).to(self.device)
                self.noise = noise
            else:
                noise = self.noise.to(self.device)
            
            latents_noisy = self.scheduler.add_noise(latents, noise, chosed_t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=chosed_t)

            noise_pred_ = self.unet(
                latent_model_input.to(self.device1),
                chosed_t.to(self.device1),
                encoder_hidden_states=[t_embeddings.to(self.device1) for t_embeddings in text_embeddings],
            ).sample.to(self.device)

        (
            sd_cross_attention_maps1,
            sd_cross_attention_maps2,
            sd_self_attention_maps,
        ) = self.get_attention_map(
            self.attention_maps,
            output_size=attention_output_size,
            token_ids=token_ids,
            average_layers=average_layers,
            train=train,
            apply_softmax=apply_softmax,
            cross_atten=cross_atten,
            beta=beta,
        )
        self.attention_maps = {}

        noise_pred_uncond, noise_pred_text = noise_pred_.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        return (
            loss,
            sd_cross_attention_maps1,
            sd_cross_attention_maps2,
            sd_self_attention_maps,
        )

    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )
            latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)
        all_attention_maps = []

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input.to(self.device1),
                        t.to(self.device1),
                        encoder_hidden_states=[t_embeddings.to(self.device1) for t_embeddings in text_embeddings],
                    )["sample"].to(self.device)

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
                if 10 <= i <= 25:
                    all_attention_maps.append(deepcopy(self.attention_maps))

        return latents, all_attention_maps

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  
        text_embeds = torch.cat(text_embeds, dim=0)
        # Text embeds -> img latents
        latents, all_attention_maps = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  

        # # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs, all_attention_maps

    def load_lora(self, state_dict_path=None):
        unet_lora_attn_procs = {}
        if state_dict_path != None:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            # print(state_dict)

        for name,value in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            unet_lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

            if state_dict_path != None:
                weights = {
                    "to_k_linear.weight": state_dict["unet." + name + ".to_k_linear.weight"],
                    "to_q_linear.weight": state_dict["unet." + name + ".to_q_linear.weight"],
                }
                unet_lora_attn_procs[name].load_state_dict(weights)
        
        self.unet.set_attn_processor(unet_lora_attn_procs)
        unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)

        self.unet = self.unet.to(self.device1)
        return unet_lora_layers