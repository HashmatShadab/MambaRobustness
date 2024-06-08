#!/bin/bash

DATA_PATH=${1:-"/path/to/imagenet/val/"}
attack_name=${2:-"pgd"}
batch_size=${3:-64}

#model_names=("resnet18" "resnet50" "vgg16_bn" "vgg19_bn" "densenet121" "densenet161" "vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224" "deit_tiny_patch16_224" "deit_small_patch16_224" "deit_base_patch16_224" "swin_tiny_patch4_window7_224" "swin_small_patch4_window7_224" "swin_base_patch4_window7_224" "vssm_tiny_v2" "vssm_small_v2" "vssm_base_v2")
model_names=("resnet50")

for model_name in "${model_names[@]}"
  do
    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 1/255 with Low Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 1  \
    --filter True --filter_size 32 --filter_preserve low --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_1_steps_20_filter_low_size_32"
    
    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 2/255 with Low Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 2  \
    --filter True --filter_size 32 --filter_preserve low --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_2_steps_20_filter_low_size_32"
    
    
    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 4/255 with Low Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 4  \
    --filter True --filter_size 32 --filter_preserve low --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_4_steps_20_filter_low_size_32"
    
    
    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 8/255 with Low Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 8  \
    --filter True --filter_size 32 --filter_preserve low --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_8_steps_20_filter_low_size_32"

    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 12/255 with Low Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 12  \
    --filter True --filter_size 32 --filter_preserve low --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_12_steps_20_filter_low_size_32"
    
    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 16/255 with Low Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 16  \
    --filter True --filter_size 32 --filter_preserve low --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_16_steps_20_filter_low_size_32"



    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 1/255 with High Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 1  \
    --filter True --filter_size 32 --filter_preserve high --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_1_steps_20_filter_high_size_32"

    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 2/255 with High Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 2  \
    --filter True --filter_size 32 --filter_preserve high --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_2_steps_20_filter_high_size_32"


    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 4/255 with High Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 4  \
    --filter True --filter_size 32 --filter_preserve high --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_4_steps_20_filter_high_size_32"


    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 8/255 with High Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 8  \
    --filter True --filter_size 32 --filter_preserve high --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_8_steps_20_filter_high_size_32"

    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 12/255 with High Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 12  \
    --filter True --filter_size 32 --filter_preserve high --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_12_steps_20_filter_high_size_32"

    echo "Craft adversarial images using PGD attack for model ${model_name}. With ${attack_name} attack at epsilon 16/255 with High Pass Filter"
    python generate_adv_images.py --data_dir ${DATA_PATH} --attack_name ${attack_name} --batch_size ${batch_size} \
    --source_model_name ${model_name} --epsilon 16  \
    --filter True --filter_size 32 --filter_preserve high --save_results_only True
    echo "The evaluation results are stored in folder AdvExamples_freq_results/${model_name}/${attack_name}_eps_16_steps_20_filter_high_size_32"

  done


