# ![Work-In-Progress Status](https://img.shields.io/badge/status-WIP-yellow) [**WORK IN PROGRESS NOTICE**] Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization]
###  [Arxiv Preprint](https://arxiv.org/abs/2412.01244),  [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Concept_Replacer_Replacing_Sensitive_Concepts_in_Diffusion_Models_via_Precision_CVPR_2025_paper.pdf)

## Installation Guide
To begin using Concept Replacer, you first need to create a virtual environment and install the dependencies using the following commands:
```
conda create -n concept_replacer
conda activate concept_replacer
pip install -r requirements.txt
```
## Concept Location
The main change of diffusers is made in diffusers/models/attention_processor.py, unet_2d_condition.py.
### Training Guide
```
python3 -m src.main --dataset_name {DATASET_NAME} \
                    --part_names  {PARTNAMES} \
                    --train_data_dir {TRAIN_DATA_DIR} \
                    --val_data_dir {VAL_DATA_DIR} \
                    --test_data_dir {TEST_DATA_DIR} \
                    --min_crop_ratio 0.6 \
                    --train_t 5 100 \
                    --epochs 200 \
                    --epoch_to_switch 100 \
                    --lr_lora 0.00005 \
                    --save_test_predictions \
                    --sd_version 2.1 \
                    --output_dir outputs \
                    --train
```

### Testing with the trained weights
```
python -m src.main --dataset {DATASET_NAME} \
                   --part_names {PARTNAMES} \
                   --checkpoint_dir {CHECKPOINT_DIR} \
                   --test_data_dir {TEST_DATA_DIR} \
                   --save_test_predictions \
                   --avg_time \
                   --output_dir ./test
```

## Concept replacing
TBD

## Acknowledgments
We thank the following contributors that our code is based on:[diffusers](https://github.com/huggingface/diffusers), [SLiMe](https://github.com/aliasgharkhani/SLiMe)

## Citing our work
The preprint can be cited as follows
```
@inproceedings{zhang2025concept,
  title={Concept replacer: Replacing sensitive concepts in diffusion models via precision localization},
  author={Zhang, Lingyun and Xie, Yu and Fu, Yanwei and Chen, Ping},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={8172--8181},
  year={2025}
}
```
