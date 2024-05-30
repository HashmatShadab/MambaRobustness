#!/bin/bash

DATA_PATH=${1:-"F:\Code\datasets\ImageNet\val"}
exp_num=${2:-1}
shuffle_size=${3:-14} # 4,8, 14, 28, 56, 112



#
 model_names=(
  "vssm_tiny_v2"
  "vssm_small_v2"
  "vssm_base_v2"
  "vit_tiny_patch16_224"
  "vit_small_patch16_224"
  "vit_base_patch16_224"
  "deit_tiny_patch16_224"
  "deit_small_patch16_224"
  "deit_base_patch16_224"
  "swin_tiny_patch4_window7_224"
  "swin_small_patch4_window7_224"
  "swin_base_patch4_window7_224"
  "resnet50"
  "vgg16_bn"
  "vgg19_bn"
  "densenet121"
  "densenet161"
  "vssm_tiny_v0"
  "vssm_small_v0"
  "vssm_base_v0"
 )

model_names=("resnet18")
directions=("1" "2" "3" "4")



 if [ $exp_num -eq 1 ]
 then

 for model_name in "${model_names[@]}"; do
   echo "model_name: $model_name"
   for direction in "${directions[@]}"; do
     echo "Running random_drop_v2_increase_forward for direction $direction with shuffle_size $shuffle_size"
     python evaluate_scanline_infodrop.py \
       --model_name $model_name \
       --test_dir "$DATA_PATH" \
       --random_drop_v2_increase_forward \
       --v2_direction $direction \
       --shuffle_size $shuffle_size $shuffle_size
   done
 done

 fi


 if [ $exp_num -eq 2 ]
 then

 for model_name in "${model_names[@]}"; do
   echo "model_name: $model_name"
   for direction in "${directions[@]}"; do
     echo "Running random_drop_v2_max_at_center for direction $direction with shuffle_size $shuffle_size"
     python evaluate_scanline_infodrop.py \
       --model_name $model_name \
       --test_dir "$DATA_PATH" \
       --random_drop_v2_max_at_center \
       --v2_direction $direction \
       --shuffle_size $shuffle_size $shuffle_size
   done
 done

 fi

 if [ $exp_num -eq 3 ]
 then

 for model_name in "${model_names[@]}"; do
   echo "model_name: $model_name"
   for direction in "${directions[@]}"; do
     echo "Running random_drop_v2_min_at_center for direction $direction with shuffle_size $shuffle_size"
     python evaluate_scanline_infodrop.py \
       --model_name $model_name \
       --test_dir "$DATA_PATH" \
       --random_drop_v2_min_at_center \
       --v2_direction $direction \
       --shuffle_size $shuffle_size $shuffle_size
   done
 done

 fi


 if [ $exp_num -eq 4 ]
 then

for model_name in "${model_names[@]}"; do
  echo "model_name: $model_name"
  for direction in "${directions[@]}"; do
    echo "Running random_drop_v3 for direction $direction with shuffle_size $shuffle_size"
     python evaluate_scanline_infodrop.py \
      --model_name $model_name \
      --test_dir "$DATA_PATH" \
      --random_drop_v3 \
      --exp_count 1 \
      --v2_direction $direction \
      --shuffle_size $shuffle_size $shuffle_size
  done
done
fi

