#!/bin/bash

DATA_PATH=${1:-"F:\Code\datasets\ImageNet\val"}
shuffle_size=${2:-14} # 4,8, 14, 28, 224
# resnet 18
python evaluate.py \
  --model_name resnet18 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size
#
## resnet 50
python evaluate.py \
  --model_name resnet50 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size


# vgg16_bn
python evaluate.py \
  --model_name vgg16_bn \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vgg19_bn
python evaluate.py \
  --model_name vgg19_bn \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# densenet121
python evaluate.py \
  --model_name densenet121 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# densenet161
python evaluate.py \
  --model_name densenet161 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vit_tiny_patch16_224
python evaluate.py \
  --model_name vit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vit_small_patch16_224
python evaluate.py \
  --model_name vit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vit_base_patch16_224
python evaluate.py \
  --model_name vit_base_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# deit_tiny_patch16_224
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# deit_small_patch16_224
python evaluate.py \
  --model_name deit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# deit_base_patch16_224
python evaluate.py \
  --model_name deit_base_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# swin_tiny_patch4_window7_224
python evaluate.py \
  --model_name swin_tiny_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# swin_small_patch4_window7_224
python evaluate.py \
  --model_name swin_small_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# swin_base_patch4_window7_224
python evaluate.py \
  --model_name swin_base_patch4_window7_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vssm_tiny_v0
python evaluate.py \
  --model_name vssm_tiny_v0 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vssm_small_v0
python evaluate.py \
  --model_name vssm_small_v0 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vssm_base_v0
python evaluate.py \
  --model_name vssm_base_v0 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size


# vssm_tiny_v2
python evaluate.py \
  --model_name vssm_tiny_v2 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vssm_small_v2
python evaluate.py \
  --model_name vssm_small_v2 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size

# vssm_base_v2
python evaluate.py \
  --model_name vssm_base_v2 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size $shuffle_size $shuffle_size
