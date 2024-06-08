## Requirements
```python
conda create -n mamba_robust

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r req.txt
cd selective_scan
pip install .
pytest
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/transformers.git

```
Download VMamba ImageNet pre-trained [weights](https://drive.google.com/drive/folders/1ceS0C1MGdOZcBNBLw4gESswarz4L54vD?usp=drive_link) and put them in `pretrained_weights` folder.

Download pre-trained weights for object detectors [(Link)](https://drive.google.com/drive/folders/1Gm_htsggYxFgYr3zVAo9-vpjjPJvcYcR?usp=drive_link) and segmentation networks
[(Link)](https://drive.google.com/drive/folders/1qbjk1B9S4Gh1XDjAq9p-sSB7C-bJ8JiN?usp=drive_link).

## A. Robustness against Adversarial attacks

### 1. White box Attacks

For crafting adversarial examples using Fast Gradient Sign Method (FGSM) at perturbation budget of 8/255, run:
```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name fgsm  --source_model_name <model_name> --epsilon 8  
```
For crafting adversarial examples using Projected Gradient Descent (PGD) at perturbation budget of 8/255 with number of attacks steps equal to 20, run:
```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name pgd  --source_model_name <model_name> --epsilon 8 --attack_steps 20 
```
Other available attacks: `bim, mifgsm, difgsm, tpgd, tifgsm, vmifgsm`


The results will be saved in  `AdvExamples_results` folder with the following structure: `AdvExamples_results/pgd_eps_{eps}_steps_{step}/{source_model_name}/accuracy.txt`


### 2. White box Frequency Attacks

#### Low Frequency Attack
For crafting adversarial examples using Projected Gradient Descent (PGD) at perturbation budget of 8/255 with number of attacks steps equal to 20, run:
```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name pgd  --source_model_name <model_name> --epsilon 8 --attack_steps 20 --filter True --filter_preserve low 
```
#### High Frequency Attack
For crafting adversarial examples using Projected Gradient Descent (PGD) at perturbation budget of 8/255 with number of attacks steps equal to 20, run:
```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name pgd  --source_model_name <model_name> --epsilon 8 --attack_steps 20 --filter True --filter_preserve high
```
The results will be saved in  `AdvExamples_freq_results` folder.

Run the below script to evaluate the robustness across different models against low and high frequency attacks at various perturbation budgets:
```python
cd  classification/
bash scripts/get_adv_freq_results.sh <DATA_PATH> <ATTACK_NAME> <BATCH_SIZE>
```


### 3. Transfer-based Black box Attacks

For evaluating transferability of adversarial examples, first save the generated adversarial examples by running:

```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name fgsm  --source_model_name <model_name> --epsilon 8 --save_results_only False  
```

The adversarial examples will be saved in  `AdvExamples` folder with the following structure: `AdvExamples/{attack_name}_eps_{eps}_steps_{step}/{source_model_name}/images_labels.pt`

Then run the below script to evaluate transferability of the generated adversarial examples across different models:

```python
cd  classification/
python inference.py --dataset imagenet_adv --data_dir <path to adversarial dataset> --batch_size <> --source_model_name <model name>
```
`--source_model_name`: name of the model on which the adversarial examples will be evaluated

Available Classification models:
1. VMamba : `vssm_tiny_v0, vssm_small_v0, vssm_base_v0, vssm_tiny_v2, vssm_small_v2, vssm_base_v2`
2. Vision Transformer: `vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224`
3. Swin Transformer: `swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224`
4. ConvNext: `convnext_tiny, convnext_small, convnext_base`
5. ResNet: `resnet18, resnet50`
6. VGG: `vgg16_bn, vgg19_bn`

Furthermore, bash scripts are provided to evaluate transferability of adversarial examples across different models:
```python
cd  classification/
# Generate adversarial examples
bash scripts/gen_adv_examples.sh <DATA_PATH> <EPSILON> <ATTACK_NAME> <BATCH_SIZE>
# Evaluate transferability of adversarial examples saved in AdvExamples folder
bash scripts/evaluate_transferability.sh <DATA_PATH> <EPSILON> <ATTACK_NAME> <BATCH_SIZE>
```


## B. Robustness against Information Drop 

### 1. Information Drop Along Scanning Lines

Run the below script to evaluate the robustness of all the models against information drop along scanning lines:
```python
cd  classification/
bash scripts/scan_line_info_drop.sh <DATA_PATH> <EXP_NUM> <PATCH_SIZE>
```
`<DATA_PATH>`: path to the dataset and <PATCH_SIZE>: number of patches the image is divided into.

`<EXP_NUM>`:
- 1:  linearly increasing the amount of information dropped in each patch along the scanning direction.
- 2:  Increasing the amount of information dropped in each patch with maximum at center of the scanning direction.
- 3:  Decreasing the amount of information dropped in each patch with maximum at center of the scanning direction.
- 4: Sequentially dropping patches along the scanning directions.

### 2. Random Patch Drop
Run the below script to evaluate the robustness of all the models against random drop of patches:
```python
cd  classification/
bash scripts/random_patch_drop.sh <DATA_PATH> <PATCH_SIZE>
```
`<DATA_PATH>`: path to the dataset and <PATCH_SIZE>: number of patches the image is divided into.

### 3. Salient Patch Drop
Run the below script to evaluate the robustness of all the models against random drop of patches:
```python
cd  classification/
bash scripts/salient_drop.sh <DATA_PATH> <PATCH_SIZE>
```
`<DATA_PATH>`: path to the dataset and <PATCH_SIZE>: number of patches the image is divided into.

### 4. Patch Shuffling
Run the below script to evaluate the robustness of all the models against random drop of patches:
```python
cd  classification/
bash scripts/shuffle_image.sh <DATA_PATH> 
```
`<DATA_PATH>`: path to the dataset


## C. Robustness against ImageNet corruptions

### Following Corrupted Datasets for Classifcation are used for evaluation:
1. ImageNet-B (Object-to-Background Compositional Changes) [(Link)](https://drive.google.com/drive/folders/1nlkwtRaL6FJeJBwcSbXhMiQ2bfqsAdmJ?usp=drive_link)
2. ImageNet-E (Attribute Editing) [(Link)](https://drive.google.com/file/d/19M1FQB8c_Mir6ermRsukTQReI-IFXeT0/view)
3. ImageNet-V2 (Synthetic Objects) [(Link)](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
4. ImageNet-A (Natural Adversarial Examples) [(Link)](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)
5. ImageNet-R (Rendition) [(Link)](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
6. ImageNet-S (Sketch) [(Link)](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA)
7. ImageNet-C (Common Corruptions) [(Link)](https://github.com/hendrycks/robustness)

### Inference on ImageNet Corrupted datasets

For evaluating on ImageNet-B, ImageNet-E, ImageNet-V2, ImageNet-A, ImageNet-R, ImageNet-S, run:
```python
cd  classification/
python inference.py --dataset <dataset name> --data_dir <path to corrupted dataset> --batch_size <> --source_model_name <model name>
```
`--dataset`: imagenet-b, imagenet-e, imagenet-v2, imagenet-a, imagenet-r, imagenet-s
`--source_model_name`: model name to use for inference

For common corruption experiment, instead of saving the corrupted datasets, the corrupted images can be generated during the evaluation by running:
```python
cd  classification/
python inference_on_imagenet_c.py --data_dir <path to imagenet validation dataset> --batch_size <> --corruption <>
```

Following `--corruption` options are available: 
1. Noise : `gaussian_noise, shot_noise, impulse_noise`
2. Blur : `defocus_blur, glass_blur, motion_blur, zoom_blur`
3. Weather : `snow, frost, fog, brightness`
4. Digital : `contrast, elastic_transform, pixelate, jpeg_compression`
5. Extra: `speckle_noise, gaussian_blur, spatter, saturate`

The script would evaluate all the models across all the severity levels of the given corruption.



## D. Robustness evaluation for Object Detection 

### Following Corrupted Datasets for Detection and Segmentation are used for evaluation:
1. COCO-O (Natural Distribution Shifts) [(Link)](https://drive.google.com/file/d/1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq/view)
2. COCO-DC (Object-to-Background Compositional Changes) [(Link)](https://drive.google.com/drive/folders/1ppLx0eyXeDS3iVs7F_k4cUOcPwCOjPWH?usp=drive_link)
3. COCO-C (Common Corruptions)
4. ADE20K-C (Common Corruptions)

Download COCO val2017 from [(here)](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) and to generate the common corruptions (COCO-C), run:

```python
python coco_corruptions.py --data_path <path to original dataset> --save_path <path to the output folder>
```
Download ADED20K  from [(here)](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and to generate the common corruptions on the validation set(ADE20K-C), run:

```python
python ade_corruptions.py --data_path <path to original dataset> --save_path <path to the output folder>
```



