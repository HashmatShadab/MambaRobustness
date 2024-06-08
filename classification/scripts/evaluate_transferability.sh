#!/bin/bash

batch_size=${1:-64}

#model_names=("resnet18" "resnet50" "vgg16_bn" "vgg19_bn" "densenet121" "densenet161" "vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224" "deit_tiny_patch16_224" "deit_small_patch16_224" "deit_base_patch16_224" "swin_tiny_patch4_window7_224" "swin_small_patch4_window7_224" "swin_base_patch4_window7_224" "vssm_tiny_v2" "vssm_small_v2" "vssm_base_v2")

model_names=("resnet50")

for data_path in AdvExamples/*/*/*.pt
  do
    echo "Evaluating transferability for adversarial examples:  ${data_path}"
  for model_name in "${model_names[@]}"
    do
      echo "Evaluating transferability for ${model_name}"
      python inference.py --dataset imagenet_adv --data_dir ${data_path}  --batch_size ${batch_size} --source_model_name ${model_name}
    done
  done



