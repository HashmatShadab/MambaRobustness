# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import logging
import time

import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader
from datasets.dataset5k import ImageNet5k

import matplotlib.pyplot as plt
import torch.nn as nn
import timm
import argparse
from PIL import Image
from imagecorruptions import corrupt
import numpy as np

import torchvision
import json
import torch
import os

try:
    from inference import  load_mamba_models
except ImportError as e:
    print(f"Error importing: {e}")

def plot_grid(w, name="test.png"):
    import matplotlib.pyplot as plt
    import torchvision
    grid_img = torchvision.utils.make_grid(w)
    # torchvision.utils.save_image(grid_img, name)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()

class ImageNet5k(torchvision.datasets.ImageFolder):

    def __init__(self, image_list="./image_list.json",
                 corruption=None, severity=None, *args, **kwargs):
        self.image_list = set(json.load(open(image_list, "r"))["images"])
        self.corruption = corruption
        self.severity = severity
        super(ImageNet5k, self).__init__(is_valid_file=self.is_valid_file, *args, **kwargs)

    def is_valid_file(self, x: str) -> bool:

        file_path = x
        # get image name
        image_name = os.path.basename(file_path)
        # get parent folder name
        folder_name = os.path.basename(os.path.dirname(file_path))

        return f"{folder_name}/{image_name}" in self.image_list

    def __getitem__(self, index):
        path, target = self.samples[index]
        pil_image = self.loader(path)
        if self.transform is not None:
            resize_pil_image = self.transform(pil_image)
            resize_array_image = np.array(resize_pil_image)
            sample = corrupt(resize_array_image, corruption_name=self.corruption, severity=self.severity)
            tensor_sample = transforms.ToTensor()(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return tensor_sample, target



class ImageNet_E(nn.Module):
    def __init__(self, data_dir, labels, transform=None):
        super(ImageNet_E, self).__init__()
        self.transform = transform
        self.imgs = []
        for img_name in os.listdir(data_dir):
            self.imgs.append([os.path.join(data_dir, img_name), labels[img_name.split('.')[0]]])

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        if torch.max(input) > 1:
            input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean.to(device=input.device)) / std.to(
            device=input.device)




def get_model(model_name=None, device="cuda"):
    # load pre-trained models

    if model_name == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True)
    elif model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
    elif model_name == 'resnet101':
        model = timm.create_model('resnet101', pretrained=True)
    elif model_name == 'vgg16_bn':
        model = timm.create_model('vgg16_bn', pretrained=True)
    elif model_name == 'vgg19_bn':
        model = timm.create_model('vgg19_bn', pretrained=True)
    elif model_name == 'densenet121':
        model = timm.create_model('densenet121', pretrained=True)
    elif model_name == 'densenet161':
        model = timm.create_model('densenet161', pretrained=True)
    elif model_name == 'vit_tiny_patch16_224':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    elif model_name == 'vit_small_patch16_224':
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
    elif model_name == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == 'deit_tiny_patch16_224':
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    elif model_name == 'deit_small_patch16_224':
        model = timm.create_model('deit_small_patch16_224', pretrained=True)
    elif model_name == 'deit_base_patch16_224':
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
    elif model_name == 'swin_tiny_patch4_window7_224':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    elif model_name == 'swin_small_patch4_window7_224':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    elif model_name == 'swin_base_patch4_window7_224':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    
    elif model_name == "convnext_tiny":
            model = timm.create_model('convnext_tiny', pretrained=True)
    elif model_name == "convnext_small":
        model = timm.create_model('convnext_small', pretrained=True)
    elif model_name == "convnext_base":
        model = timm.create_model('convnext_base', pretrained=True)
        
    elif model_name == 'vssm_tiny_v0':
    
        model = load_mamba_models('vssm_tiny_v0')
        ckpt = torch.load('../pretrained_weights/vmamba_tiny_e292.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    elif model_name == 'vssm_small_v0':

        model = load_mamba_models('vssm_small_v0')
        ckpt = torch.load('../pretrained_weights/vmamba_small_e238_ema.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    elif model_name == 'vssm_base_v0':

        model = load_mamba_models('vssm_base_v0')
        ckpt = torch.load('../pretrained_weights/vssmbase_dp06_ckpt_epoch_241.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    
    elif model_name == 'vssm_tiny_v2':
    
        model = load_mamba_models('vssm_tiny_v2')
        ckpt = torch.load('../pretrained_weights/vssm_tiny_0230_ckpt_epoch_262.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    elif model_name == 'vssm_small_v2':

        model = load_mamba_models('vssm_small_v2')
        ckpt = torch.load('../pretrained_weights/vssm_small_0229_ckpt_epoch_222.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    elif model_name == 'vssm_base_v2':

        model = load_mamba_models('vssm_base_v2')
        ckpt = torch.load('../pretrained_weights/vssm_base_0229_ckpt_epoch_237.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)

    if model_name not in ['vssm_tiny_v0', 'vssm_small_v0', 'vssm_base_v0', 'vssm_tiny_v2', 'vssm_small_v2', 'vssm_base_v2']:
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    model = nn.Sequential(Normalize(mean, std), model)

    return model.eval().to(device), mean, std



def plot_grid(w, save=False, name="grid.png"):
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    if save:
        plt.savefig(name)
    plt.show()


def load_pretrained_ema(pretrained_path, model, logger):
    logger.info(f"==============> Loading weight {pretrained_path} for fine-tuning......")
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{pretrained_path}'")
    else:
        logger.warning(f"No 'model' found in {pretrained_path}! ")

    del checkpoint


@torch.no_grad()
def validate(val_loader, model, logger):
    model.eval()  # set the model to evaluation mode

    batch_time = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end = time.time()

    for idx, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)  # Move images and labels to GPU

        # with torch.cuda.amp.autocast(enabled=True):
        outputs = model(images)
        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 10 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg


def get_args():
    parser = argparse.ArgumentParser(description='Transferability test')
    parser.add_argument('--data_dir', help='path to ImageNet dataset',
                        default=r'F:\Code\datasets\ImageNet\val')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--corruption', default='gaussian_noise')

    args = parser.parse_args()

    return args

"""
corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate,
                    jpeg_compression, speckle_noise, gaussian_blur, spatter,
                    saturate)
"""

if __name__ == "__main__":
    args = get_args()

    data_dir = args.data_dir
    # get parent directory
    parent_dir = os.path.dirname(data_dir)
    # save log path
    model_names = (
        "resnet50",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        "vssm_tiny_v0",
        "vssm_small_v0",
        "vssm_base_v0",
        "vssm_tiny_v2",
        "vssm_small_v2",
        "vssm_base_v2",
    )


    parent_dir = "./imagenet_c_logs"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    log_dir = os.path.join(parent_dir, f"{args.corruption}_eval_imagent_c.log")

    logging.basicConfig(filename=log_dir, filemode="a",
                        format="%(name)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add console handler to logger
    logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ine_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        ])
    for severity in range(1, 6):
        dataset = ImageNet5k(root=args.data_dir, corruption=args.corruption, severity=severity, transform=ine_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        images = []
        labels = []
        for idx, (image, label) in enumerate(dataloader):
            images.append(image)
            labels.append(label)
            print(idx)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        corrupted_dataset = torch.utils.data.TensorDataset(images, labels)
        corrupted_dataloader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=args.batch_size, shuffle=False)



        for model_name in model_names:
            print(f"Validating {model_name} {args.corruption} {severity}")
            model, _, _ = get_model(model_name, device)
            acc = validate(corrupted_dataloader, model, logger)
            logger.info(f"Accuracy {model_name} {args.corruption} {severity}: {acc}")




