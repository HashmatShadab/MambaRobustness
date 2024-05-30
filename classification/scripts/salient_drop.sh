#!/bin/bash

DATA_PATH=${1:-"F:\Code\datasets\ImageNet\val"}

# resnet 18
python evaluate.py \
  --model_name resnet18 \
  --test_dir "$DATA_PATH" \
  --dino
  
# resnet 50
python evaluate.py \
  --model_name resnet50 \
  --test_dir "$DATA_PATH" \
  --dino

# vgg16_bn
python evaluate.py \
  --model_name vgg16_bn \
  --test_dir "$DATA_PATH" \
  --dino

# vgg19_bn
python evaluate.py \
  --model_name vgg19_bn \
  --test_dir "$DATA_PATH" \
  --dino

# densenet121
python evaluate.py \
  --model_name densenet121 \
  --test_dir "$DATA_PATH" \
  --dino

# densenet161
python evaluate.py \
  --model_name densenet161 \
  --test_dir "$DATA_PATH" \
  --dino

# vit_tiny_patch16_224
python evaluate.py \
  --model_name vit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino

# vit_small_patch16_224
python evaluate.py \
  --model_name vit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vit_base_patch16_224
python evaluate.py \
  --model_name vit_base_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino

# deit_tiny_patch16_224
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino

# deit_small_patch16_224
python evaluate.py \
  --model_name deit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino

# deit_base_patch16_224
python evaluate.py \
  --model_name deit_base_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino

 swin_tiny_patch4_window7_224
python evaluate.py \
  --model_name swin_tiny_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --dino

# swin_small_patch4_window7_224
python evaluate.py \
  --model_name swin_small_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --dino

# swin_base_patch4_window7_224
python evaluate.py \
  --model_name swin_base_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vssm_tiny_v0 
python evaluate.py \
  --model_name vssm_tiny_v0 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vssm_small_v0
python evaluate.py \
  --model_name vssm_small_v0 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vssm_base_v0
python evaluate.py \
  --model_name vssm_base_v0 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vssm_tiny_v2
python evaluate.py \
  --model_name vssm_tiny_v2 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vssm_small_v2
python evaluate.py \
  --model_name vssm_small_v2 \
  --test_dir "$DATA_PATH" \
  --dino
  
# vssm_base_v2
python evaluate.py \
  --model_name vssm_base_v2 \
  --test_dir "$DATA_PATH" \
  --dino