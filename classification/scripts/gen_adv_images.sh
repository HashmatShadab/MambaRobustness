#!/bin/bash

DATA_PATH=${1:-"/path/to/imagenet/root/folder"}
epsilon=${2:-8}
attack_name=${3:-"pgd"}
batch_size=${4:-64}
filter=${5:-false}
filter_preserve=${6:-"low"}
save_results_only=${7:-false}

#model_names=("resnet18" "resnet50" "vgg16_bn" "vgg19_bn" "densenet121" "densenet161" "vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224" "deit_tiny_patch16_224" "deit_small_patch16_224" "deit_base_patch16_224" "swin_tiny_patch4_window7_224" "swin_small_patch4_window7_224" "swin_base_patch4_window7_224" "vssm_tiny_v2" "vssm_small_v2" "vssm_base_v2")

model_names=("resnet50")

for model_name in "${model_names[@]}"
  do
    echo "Craft adversarial images using PGD attack with perturbation budget ${epsilon} for model ${model_name}."

    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name $attack_name --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon ${epsilon}  \
    --filter false --filter_preserve ${filter_preserve} --save_results_only ${save_results_only}

    if [ "$attack_name" == "pgd" ]; then
         echo "Adversarial images crafted for model ${model_name}. The adversarial images are stored in folder AdvExamples/${attack_name}_eps_${epsilon}_steps_20/${model_name}"

    elif [ "$attack_name" == "fgsm" ]; then
          echo "Adversarial images crafted for model ${model_name}. The adversarial images are stored in folder AdvExamples/${attack_name}_eps_${epsilon}_steps_1/${model_name}"

    else
         echo "Adversarial images crafted for model ${model_name}. The adversarial images are stored in folder AdvExamples/${attack_name}_eps_${epsilon}_steps_20/${model_name}"
    fi

  done


# For example bash gen_adv_images.sh /path/to/imagenet/val/ 8 pgd 64 for standard PGD attack
# For example bash gen_adv_images.sh /path/to/imagenet/val/ 8 fgsm 64 for standard FGSM attack
# For example bash gen_adv_images.sh /path/to/imagenet/val/ 8 pgd 64 true low for PGD attack with low-pass filter
# For example bash gen_adv_images.sh /path/to/imagenet/val/ 8 fgsm 64 true low for FGSM attack with low-pass filter
# set save_results_only to true to only evaluate the model on the adversarial attack without saving the adversarial images