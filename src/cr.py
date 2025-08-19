import os

import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from src.stable_difusion import StableDiffusion
from src.utils import (
    calculate_iou,
    get_crops_coords,
    generate_distinct_colors,
    get_colored_segmentation,
    get_colored_segmentation_with_blur,
    get_boundry_and_eroded_mask,
)
import gc
from PIL import Image
import numpy as np
import csv
from collections import defaultdict
import re
import itertools
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin


import torch

def dice_loss(A, B, smooth=1e-6):
    # Apply sigmoid activation if A contains logits
    # A = torch.sigmoid(A)
    
    # Flatten the tensors (keep batch and class dims if needed)
    A_flat = A.view(A.size(0), -1)  # Flatten along spatial dimensions
    B_flat = B.view(B.size(0), -1)
    
    # Compute intersection
    intersection = (A_flat * B_flat).sum(dim=1)
    
    # Compute Dice loss for each class
    dice_score = (2. * intersection + smooth) / (A_flat.sum(dim=1) + B_flat.sum(dim=1) + smooth)
    
    # Dice loss is 1 - dice_score
    dice_loss = 1 - dice_score.mean()  # Mean across classes
    
    return dice_loss

class CR(pl.LightningModule):
    def __init__(self, config, learning_rate=0.001):
        super().__init__()
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.max_val_iou = 0
        self.val_ious = []
        self.all_iou_res = []
        self.cross_atten = self.config.cross_atten
        self.output_dir = self.config.output_dir
        self.beta = torch.nn.Parameter(torch.full((1, len(self.config.part_names), 1, 1), 0.3))  # C = number of parts

        if self.config.checkpoint_dir == None:
            self.checkpoint_dir = self.config.output_dir
        else:
            self.checkpoint_dir = self.config.checkpoint_dir

        self.stable_diffusion = StableDiffusion(
            sd_version=self.config.sd_version,
            attention_layers_to_use=config.attention_layers_to_use,
        )
        self.unet_lora_layers = self.stable_diffusion.load_lora()
        self.num_parts = len(self.config.part_names)

        self.prepare_text_embeddings()
        del self.stable_diffusion.tokenizer
        del self.stable_diffusion.text_encoder
        torch.cuda.empty_cache()

        self.embeddings_to_optimize_low, self.embeddings_to_optimize_high = [], []

        if self.config.train:
            for i in range(1, self.num_parts):
                embedding = self.text_embedding[:, i : i + 1].detach().clone()
                embedding.requires_grad_(True)
                self.embeddings_to_optimize_low.append(embedding)
            
            for i in range(1, self.num_parts):
                embedding = self.text_embedding[:, i : i + 1].detach().clone()
                embedding.requires_grad_(False)
                self.embeddings_to_optimize_high.append(embedding)

        self.token_ids = list(range(self.num_parts))

    def prepare_text_embeddings(self):
        if self.config.text_prompt is None:
            text_prompt = " ".join(["part" for _ in range(self.num_parts)])
        else:
            text_prompt = self.config.text_prompt
        (
            self.uncond_embedding,
            self.text_embedding,
        ) = self.stable_diffusion.get_text_embeds(text_prompt, "")

    def on_fit_start(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.stable_diffusion.setup(self.device)
        self.uncond_embedding, self.text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)


    def training_step(self, batch, batch_idx):
        image, mask = batch
        num_pixels = torch.zeros(self.num_parts, dtype=torch.int64).to(self.device)
        values, counts = torch.unique(mask, return_counts=True)
        num_pixels[values.type(torch.int64)] = counts.type(torch.int64)
        num_pixels[0] = 0
        pixel_weights = torch.where(num_pixels > 0, num_pixels.sum() / (num_pixels + 1e-6), 0)
        pixel_weights[0] = 1
        mask = mask[0]
        t_embeddings = []
        for embeddings in [self.embeddings_to_optimize_low, self.embeddings_to_optimize_high]:
            text_embedding = torch.cat(
                [
                    self.text_embedding[:, 0:1],
                    *list(map(lambda x: x.to(self.device), embeddings)),
                    self.text_embedding[:, 1 + len(embeddings):],
                ],
                dim=1,
            )
            t_embedding = torch.cat([self.uncond_embedding, text_embedding])
            t_embeddings.append(t_embedding)
        
        if (self.current_epoch == 0) and (batch_idx == 0):
            t_embeddings[1] = t_embeddings[0]
            for embeddings in self.embeddings_to_optimize_high:
                embeddings.requires_grad_(False)
            self.beta.requires_grad_(False)
            parameters = [{"params": self.embeddings_to_optimize_low, "lr": self.config.lr},  {"params": self.unet_lora_layers.parameters(), "lr": self.config.lr_lora}]
            self.trainer.optimizers = [getattr(optim, self.config.optimizer)(
                parameters,
                lr=self.config.lr,
            )]
            self.cross_atten = "low"

        if (self.current_epoch == self.config.epoch_to_switch) and (batch_idx == 0):
            self.load_weights()
            
            for embeddings in self.embeddings_to_optimize_low:
                embeddings.requires_grad_(False)
            for parameters in self.unet_lora_layers.parameters():
                parameters.requires_grad_(False)
            for embeddings in self.embeddings_to_optimize_high:
                embeddings.requires_grad_(True)
            self.beta.requires_grad_(True)
            parameters = [{"params": self.embeddings_to_optimize_high, "lr": self.config.lr}, {"params": self.beta, "lr": 5e-4}]
            self.trainer.optimizers = [getattr(optim, self.config.optimizer)(
                parameters,
                lr=self.config.lr,
            )]
            self.cross_atten = "merge"
            
        (
            sd_loss,
            _,
            sd_cross_attention_maps2,
            sd_self_attention_maps,
        ) = self.stable_diffusion.train_step(
            t_embeddings,
            image,
            t=torch.tensor(self.config.train_t),
            attention_output_size=self.config.train_mask_size,
            token_ids=self.token_ids,
            train=True,
            average_layers=True,
            apply_softmax=False,
            cross_atten=self.cross_atten,
            beta=self.beta,
        )

        loss1 = F.cross_entropy(
            sd_cross_attention_maps2[None, ...],
            mask[None, ...].type(torch.long),
            weight=pixel_weights,
        )
        sd_cross_attention_maps2 = sd_cross_attention_maps2.softmax(dim=0)

        small_sd_cross_attention_maps2 = F.interpolate(
            sd_cross_attention_maps2[None, ...], 64, mode="bilinear", antialias=True
        )[0]
        self_attention_map = (
            sd_self_attention_maps[None, ...]
            * small_sd_cross_attention_maps2.flatten(1, 2)[..., None, None]
        ).sum(dim=1)
        one_shot_mask = (
            torch.zeros(
                self.num_parts,
                mask.shape[0],
                mask.shape[1],
            )
            .to(mask.device)
            .scatter_(0, mask.unsqueeze(0).type(torch.int64), 1.0)
        )

        loss2 = F.mse_loss(self_attention_map, one_shot_mask) * self.num_parts
        loss3 = dice_loss(self_attention_map, one_shot_mask)

        sd_self_attention_maps = None
        small_sd_cross_attention_maps2 = None
        self_attention_map = None

        loss = (
            loss1
            + self.config.sd_loss_coef * sd_loss
            + self.config.self_attention_loss_coef * loss2
            + loss3
        )

        self.test_t_embedding = t_embeddings
        final_mask = self.get_patched_masks(
            image,
            self.config.train_mask_size,
            self.config.test_t,
        )

        sd_cross_attention_maps2 = None
        ious = []
        for idx, part_name in enumerate(self.config.part_names):
            part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"train {part_name} iou", iou, on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)

        self.log("loss2", loss2.detach().cpu(), on_step=True, sync_dist=True)
        self.log("sd_loss", sd_loss.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss1", loss1.detach().cpu(), on_step=True, sync_dist=True)
        self.log("train mean iou", mean_iou.cpu(), on_step=True, sync_dist=True)
        self.log("loss", loss.detach().cpu(), on_step=True, sync_dist=True)

        return loss

    def get_patched_masks(self, image, output_size, time, apply_argmax=True):
        crops_coords = get_crops_coords(
            image.shape[2:],
            self.config.patch_size,
            self.config.num_patchs_per_side,
        )

        final_attention_map = torch.zeros(
            self.num_parts,
            output_size,
            output_size,
        ).to(self.device)

        aux_attention_map = (
            torch.zeros(
                self.num_parts,
                output_size,
                output_size,
                dtype=torch.uint8,
            )
            + 1e-7
        ).to(self.device)

        ratio = 512 // output_size
        mask_patch_size = self.config.patch_size // ratio
        for crop_coord in crops_coords:
            y_start, y_end, x_start, x_end = crop_coord
            mask_y_start, mask_y_end, mask_x_start, mask_x_end = (
                y_start // ratio,
                y_end // ratio,
                x_start // ratio,
                x_end // ratio,
            )
            cropped_image = image[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                (
                    _,
                    _,
                    sd_cross_attention_maps2,
                    sd_self_attention_maps,
                ) = self.stable_diffusion.train_step(
                    self.test_t_embedding,
                    cropped_image,
                    t=torch.tensor(time),
                    generate_new_noise=True,
                    attention_output_size=64,
                    token_ids=self.token_ids,
                    train=False,
                    cross_atten=self.cross_atten,
                    beta=self.beta,
                )

                sd_cross_attention_maps2 = sd_cross_attention_maps2.flatten(1, 2)

                max_values = sd_cross_attention_maps2.max(dim=1).values
                min_values = sd_cross_attention_maps2.min(dim=1).values
                passed_indices = torch.where(max_values >= self.config.patch_threshold)[
                    0
                ]
                if len(passed_indices) > 0:
                    sd_cross_attention_maps2 = sd_cross_attention_maps2[passed_indices]
                    sd_cross_attention_maps2[0] = torch.where(
                        sd_cross_attention_maps2[0]
                        > sd_cross_attention_maps2[0].mean(),
                        sd_cross_attention_maps2[0],
                        0,
                    )
                    for idx, mask_id in enumerate(passed_indices):
                        avg_self_attention_map = (
                            sd_cross_attention_maps2[idx][..., None, None]
                            * sd_self_attention_maps
                        ).sum(dim=0)
                        avg_self_attention_map = F.interpolate(
                            avg_self_attention_map[None, None, ...],
                            mask_patch_size,
                            mode="bilinear",
                            antialias=True,
                        )[0, 0]

                        avg_self_attention_map_min = avg_self_attention_map.min()
                        avg_self_attention_map_max = avg_self_attention_map.max()
                        coef = (
                            avg_self_attention_map_max - avg_self_attention_map_min
                        ) / (max_values[mask_id] - min_values[mask_id])
                        if torch.isnan(coef) or coef == 0:
                            coef = 1e-7
                        final_attention_map[
                            mask_id,
                            mask_y_start:mask_y_end,
                            mask_x_start:mask_x_end,
                        ] += (avg_self_attention_map / coef) + (
                            min_values[mask_id] - avg_self_attention_map_min / coef
                        )
                        aux_attention_map[
                            mask_id,
                            mask_y_start:mask_y_end,
                            mask_x_start:mask_x_end,
                        ] += torch.ones_like(avg_self_attention_map, dtype=torch.uint8)

        final_attention_map /= aux_attention_map
        if apply_argmax:
            final_attention_map = final_attention_map.argmax(0)
        return final_attention_map

    def on_validation_start(self):
        self.test_t_embedding = []
        for embeddings in [self.embeddings_to_optimize_low, self.embeddings_to_optimize_high]:
            text_embedding = torch.cat(
                [
                    self.text_embedding[:, 0:1],
                    *list(map(lambda x: x.to(self.device), embeddings)),
                    self.text_embedding[:, 1 + len(embeddings):],
                ],
                dim=1,
            )
            t_embedding = torch.cat([self.uncond_embedding, text_embedding])
            self.test_t_embedding.append(t_embedding)

        if self.current_epoch < self.config.epoch_to_switch:
            self.test_t_embedding[1] = self.test_t_embedding[0]
        

    def on_validation_epoch_start(self):
        self.val_ious = []

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask[0]
        final_mask = self.get_patched_masks(
            image,
            self.config.test_mask_size,
            self.config.test_t
        )
        ious = []
        for idx, part_name in enumerate(self.config.part_names):
            part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"val {part_name} iou", iou.cpu(), on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)
        self.val_ious.append(mean_iou)
        self.log("val mean iou", mean_iou.cpu(), on_step=True, sync_dist=True)
        return torch.tensor(0.0)

    def on_validation_epoch_end(self):
        epoch_mean_iou = sum(self.val_ious) / len(self.val_ious)

        if epoch_mean_iou >= self.max_val_iou:
            self.max_val_iou = epoch_mean_iou
            if self.current_epoch < self.config.epoch_to_switch:
                for i, embedding in enumerate(self.embeddings_to_optimize_low):
                    torch.save(
                        embedding,
                        os.path.join(self.output_dir, f"embedding_0_{i}.pth"),
                    )
                LoraLoaderMixin.save_lora_weights(
                    save_directory=self.checkpoint_dir,
                    unet_lora_layers=self.unet_lora_layers
                )
            else:
                for i, embedding in enumerate(self.embeddings_to_optimize_high):
                    torch.save(
                        embedding,
                        os.path.join(self.output_dir, f"embedding_1_{i}.pth"),
                    )
                torch.save(self.beta, os.path.join(self.checkpoint_dir, "beta.pth"))

            
        gc.collect()

    def load_weights(self):
        print("Loading weights for unet and embeddings...")
        self.stable_diffusion.load_lora(
            state_dict_path=os.path.join(self.checkpoint_dir, "pytorch_lora_weights.bin")
        )

        embeddings_to_optimize_low, embeddings_to_optimize_high = [], []
        for j in range(self.num_parts - 1):
            embedding = torch.load(
                os.path.join(self.checkpoint_dir, f"embedding_0_{j}.pth")
            )
            embeddings_to_optimize_low.append(embedding)

        if (self.cross_atten == "low") or (not os.path.exists(os.path.join(self.checkpoint_dir, f"embedding_1_0.pth"))):
            print("loading low as high ..." )
            embeddings_to_optimize_high = [embedding.detach().clone() for embedding in embeddings_to_optimize_low]
        else:
            print("loading high part embeddings and beta..." )
            for j in range(self.num_parts - 1):
                embedding = torch.load(
                    os.path.join(self.checkpoint_dir, f"embedding_1_{j}.pth")
                )
                embeddings_to_optimize_high.append(embedding)

            self.beta = torch.load(
                    os.path.join(self.checkpoint_dir, f"beta.pth")
                )
        
        self.embeddings_to_optimize_low = embeddings_to_optimize_low
        self.embeddings_to_optimize_high = embeddings_to_optimize_high

    def on_test_start(self) -> None:
        self.test_t_embedding = []
        self.stable_diffusion.setup(self.device)
        uncond_embedding, text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)
        self.stable_diffusion.change_hooks(
            attention_layers_to_use=self.config.attention_layers_to_use
        )  # detach attention layers
        
        embeddings_to_optimize = []
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.config.output_dir

        self.load_weights()

        # print(self.text_embedding.device, embeddings_to_optimize[0].device,self.device)
        for embeddings in [self.embeddings_to_optimize_low, self.embeddings_to_optimize_high]:
            text_embedding = torch.cat(
                [
                    text_embedding[:, 0:1],
                    *list(map(lambda x: x.to(self.device), embeddings)),
                    text_embedding[:, 1 + len(embeddings):],
                ],
                dim=1,
            )
            t_embedding = torch.cat([uncond_embedding, text_embedding])
            self.test_t_embedding.append(t_embedding)

        if self.config.save_test_predictions:
            self.distinct_colors = generate_distinct_colors(self.num_parts - 1)
            self.test_results_dir = os.path.join(
                self.config.output_dir,
                "test_results",
                self.logger.log_dir.split("/")[-1],
            )
            os.makedirs(self.test_results_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask_provided = not torch.all(mask == 0)
        
        mask = mask[0]
        if self.config.avg_time:
            avg_final_masks = []
            for time in [10.0909, 50.4545, 100.9091, 151.3636]:
                final_mask = self.get_patched_masks(
                    image,
                    self.config.test_mask_size,
                    [time],
                    apply_argmax=False,
                )
                avg_final_masks.append(final_mask)
            final_mask = torch.stack(avg_final_masks).mean(0).argmax(0)
        else:
            final_mask = self.get_patched_masks(
                image,
                self.config.test_mask_size,
                self.config.test_t
            )

        if self.config.save_test_predictions:
            eroded_final_mask, final_mask_boundary = get_boundry_and_eroded_mask(
                final_mask.cpu()
            )
            colored_image = get_colored_segmentation_with_blur(
                torch.tensor(eroded_final_mask),
                torch.tensor(final_mask_boundary),
                image[0].cpu(),
                self.distinct_colors,
            )
            for i in range(image.shape[0]):
                Image.fromarray((255 * colored_image).type(torch.uint8).numpy()).save(
                    os.path.join(
                        self.test_results_dir, f"{batch_idx * image.shape[0] + i}.png"
                    )
                )

        if mask_provided:
            iou_results = {}
            for idx, part_name in enumerate(self.config.part_names):
                part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
                if torch.all(part_mask == 0):
                    continue
                
                iou = calculate_iou(
                    torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
                )
                self.log(
                    f"{part_name}", iou.cpu().item(), on_step=True, sync_dist=True
                )
                iou_results[part_name] = iou.cpu().item()
            
            self.all_iou_res.append(iou_results)
            
        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        
        if len(self.all_iou_res) == 0:
            return
        
        csv_file = os.path.join(self.test_results_dir,'output.csv')

        # Writing to a CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.config.part_names)
            writer.writeheader()  # Writing the header
            writer.writerows(self.all_iou_res)  # Writing the rows

        sums = defaultdict(float)
        counts = defaultdict(int)
        # Sum the values and count the occurrences for each column
        for entry in self.all_iou_res:
            for key, value in entry.items():
                sums[key] += value
                counts[key] += 1

        # Calculate the mean for each column
        means = {key: sums[key] / counts[key] for key in sums}
        print("Means:",means)

        # Get the sum of all values
        total_sum = sum(means.values())

        # Get the number of values
        num_values = len(means)

        # Calculate the mean
        mean_value = total_sum / num_values

        print("Total Mean:", mean_value)

        print("max val mean iou: ", self.max_val_iou)
        print("self.beta:", self.beta)

    def configure_optimizers(self):
        parameters = [{"params": self.embeddings_to_optimize_low, "lr": self.config.lr}, {"params": self.unet_lora_layers.parameters(), "lr": self.config.lr_lora}]
        optimizer = getattr(optim, self.config.optimizer)(
            parameters,
            lr=self.config.lr,
        )

        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
