#!/bin/bash

DATA_PATH=${1:-"F:\Code\datasets\ImageNet\val"}

# resnet 18
python evaluate.py \
  --model_name resnet18 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# resnet 50
python evaluate.py \
  --model_name resnet50 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16


# vgg16_bn
python evaluate.py \
  --model_name vgg16_bn \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16


# vgg19_bn
python evaluate.py \
  --model_name vgg19_bn \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16


# densenet121
python evaluate.py \
  --model_name densenet121 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# densenet161
python evaluate.py \
  --model_name densenet161 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# vit_tiny_patch16_224
python evaluate.py \
  --model_name vit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# vit_small_patch16_224
python evaluate.py \
  --model_name vit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# vit_base_patch16_224
python evaluate.py \
  --model_name vit_base_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# deit_tiny_patch16_224
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# deit_small_patch16_224
python evaluate.py \
  --model_name deit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# deit_base_patch16_224
python evaluate.py \
  --model_name deit_base_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# swin_tiny_patch4_window7_224
python evaluate.py \
  --model_name swin_tiny_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# swin_small_patch4_window7_224
python evaluate.py \
  --model_name swin_small_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# swin_base_patch4_window7_224
python evaluate.py \
  --model_name swin_base_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16

# vssm_tiny_v0
python evaluate.py \
  --model_name vssm_tiny_v0 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16
  
# vssm_small_v0
python evaluate.py \
  --model_name vssm_small_v0 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16
  
# vssm_base_v0
python evaluate.py \
  --model_name vssm_base_v0 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16
  
# vssm_tiny_v2
python evaluate.py \
  --model_name vssm_tiny_v2 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16
  
# vssm_small_v2
python evaluate.py \
  --model_name vssm_small_v2 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16
  
# vssm_base_v2
python evaluate.py \
  --model_name vssm_base_v2 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16
  
  

