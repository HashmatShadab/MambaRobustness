# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import logging
import os
import time

import torch
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from dataset import ImageNet5k

# from attacks import PGD
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import timm
import argparse
from PIL import Image


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
        return  img, label

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


def load_mamba_models(model_name):
    from models.vmamba import VSSM

    if model_name == "vssm_small_v0":
        model = VSSM(
            in_chans=3,
            patch_size=4,
            num_classes=1000,
            depths=[2, 2, 27, 2],
            dims=96,
            # ===================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.0,
            shared_ssm=False,
            softmax_version=False,
            # ================
            drop_rate=0.0,
            drop_path_rate=0.3,
            mlp_ratio=0.0,
            patch_norm=True,
            # =================
            downsample_version='v1',
            use_checkpoint=False
        )
    elif model_name == "vssm_small_v2":
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 15, 2],
            dims=96,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.3,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
        )

    elif model_name == "vssm_tiny_v0":
        model = VSSM(
            in_chans=3,
            patch_size=4,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=96,
            # ===================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.0,
            shared_ssm=False,
            softmax_version=False,
            # ================
            drop_rate=0.0,
            drop_path_rate=0.2,
            mlp_ratio=0.0,
            patch_norm=True,
            # =================
            downsample_version='v1',
            use_checkpoint=False
        )
    elif model_name == "vssm_tiny_v2":
            model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 5, 2],
            dims=96,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto", ####
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",   #####
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,  ####
            # ===================
            drop_path_rate=0.2,
            patch_norm=True,   #####
            norm_layer="ln",   #####
            downsample_version="v3",
            patchembed_version="v2", ###gmlp=false
            use_checkpoint=False,
        )

    elif model_name == "vssm_base_v0":

        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 27, 2],
            dims=128,
            # ===================
            ssm_d_state=16, # ssm_d_state 16, ssm_rank_ration 2.0
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto", # ssm_dt_rank, ssm_act_layer silu, ssm_cov 3, ssm_cov_bias True, ssm_drop_rate 0.0,ssm_init v0, forward_tpe v2, mlp_ratio 0.0,
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",


            mlp_ratio=0.0,
            mlp_act_layer="gilu",
            mlp_drop_rate=0.0,

            drop_path_rate=0.0,
            patch_norm=True,
            # norm_layer="ln",
            downsample_version='v1',
            patchembed_version="v1",
            use_checkpoint=False

            # mlp_act_layer gilu, mlp_drop_rate 0.0, norm_layer ln, patchm,ebed version v1


        )
    elif model_name == "vssm_base_v2":
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 15, 2],
            dims=128,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.6,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


    return model


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
    plt.imshow(grid_img.permute(1,2,0).cpu())
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
            logger.info(
                f'Test: [{idx}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg


def get_args():
    parser = argparse.ArgumentParser(description='Transferability test')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name', choices=['imagenet5k', 'imagenet_adv', 'imagenet_full', 'imagenet-e', 'imagenet-b', 'imagenet-a', 'imagenet-r', 'imagenet-s', 'imagenet-c', 'imagenet-v2'])
    parser.add_argument('--data_dir', help='path to ImageNet dataset', default=r'F:\Code\datasets\ImageNet\val')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--source_model_name', default='resnet18')

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()


    data_dir = args.data_dir
    # get parent directory
    parent_dir = os.path.dirname(data_dir)
    # save log path
    if args.dataset == 'imagenet5k':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet5k.log")
    elif args.dataset == 'imagenet_full':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_full.log")
    elif args.dataset == 'imagenet_adv':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet_adv.log")
    elif args.dataset == 'imagenet-e':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet_e.log")
    elif args.dataset == 'imagenet-b':
        log_dir = os.path.join(data_dir, f"{args.source_model_name}_eval_imagenet_b.log")
    elif args.dataset == 'imagenet-a':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet_a.log")
    elif args.dataset == 'imagenet-r':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet_r.log")
    elif args.dataset == 'imagenet-s':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet_s.log")
    elif args.dataset == 'imagenet-c':
        log_dir = os.path.join(data_dir, f"{args.source_model_name}_eval_imagenet_c.log")
    elif args.dataset == 'imagenet-v2':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval_imagenet_v2.log")
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset}")


    logging.basicConfig(filename=log_dir, filemode="a",
                        format="%(name)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add console handler to logger
    logger.addHandler(ch)

    # log the path where the log is saved
    logger.info(f"Log file saved at: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the model
    model, _,_ = get_model(args.source_model_name, device)

    ine_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            ])

    # Load the dataset
    if args.dataset == 'imagenet5k' or args.dataset == 'imagenet_adv':
        if args.dataset == 'imagenet_adv':
            data = torch.load(args.data_dir)
            dataset = torch.utils.data.TensorDataset(data[0], data[1])
        else:
            dataset = ImageNet5k(root=args.data_dir, transform = ine_transform )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(dataloader, model, logger)
        logger.info(f"Accuracy: {acc}")

    elif args.dataset == 'imagenet_full':

        from imagenet_dataset import ImageFolder as ImageNet_Dataset
        data_dir = args.data_dir

        dataset = ImageNet_Dataset(data_dir, transform=ine_transform)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(dataloader, model, logger)
        logger.info(f"Accuracy: {acc}")

    if args.dataset == "imagenet-s":
        from imagenet_dataset import ImageFolder as ImageNetS_Dataset

        data_dir = args.data_dir

        dataset = ImageNetS_Dataset(data_dir, transform=ine_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(data_loader, model, logger)
        logger.info(f"Accuracy: {acc}")

    if args.dataset == "imagenet-v2":
        from imagenet_v2_dataset import ImageFolder as ImageNetv2_Dataset

        data_dir = args.data_dir

        dataset = ImageNetv2_Dataset(data_dir, transform=ine_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(data_loader, model, logger)
        logger.info(f"Accuracy: {acc}")

    if args.dataset == "imagenet-a":
        from imagenet_dataset import ImageFolder as ImageNetA_Dataset

        data_dir = args.data_dir

        dataset = ImageNetA_Dataset(data_dir, transform=ine_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(data_loader, model, logger)
        logger.info(f"Accuracy: {acc}")

    if args.dataset == "imagenet-r":
        from imagenet_dataset import ImageFolder as ImageNetR_Dataset

        data_dir = args.data_dir

        dataset = ImageNetR_Dataset(data_dir, transform=ine_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(data_loader, model, logger)
        logger.info(f"Accuracy: {acc}")


    if args.dataset == 'imagenet-b':
        from imagenet_b_dataset import ImageFolder as ImageNetB_Dataset

        data_dir = args.data_dir

        for kind in ['Original', 'BliP_Caption', 'Class_Name', 'Color', 'Texture', 'Adversarial']:
            img_root = os.path.join(data_dir, kind)
            dataset = ImageNetB_Dataset(img_root, transform=ine_transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            acc = validate(data_loader, model, logger)
            logger.info(f"Accuracy: {kind}: {acc}")

    if args.dataset == "imagenet-c":
        from imagenet_dataset import ImageFolder as ImageNetC_Dataset

        data_dir = args.data_dir

        for kind in ["1", "2", "3", "4", "5"]:
            img_root = os.path.join(data_dir, kind)
            dataset = ImageNetC_Dataset(img_root, transform=ine_transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            acc = validate(data_loader, model, logger)
            logger.info(f"Accuracy: {kind}: {acc}")

    elif args.dataset == "imagenet-e":

        ine_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            ])
        data_dir = args.data_dir
        labels = {}
        for line in open(os.path.join(data_dir, 'labels.txt')):
            img_name, label = line.strip().split('\t')
            labels[img_name.split('.')[0]] = int(label)

        for kind in ['rec', '20_smooth', '20_hard', '20_adv', 'random_bg', '01', '008', '005', 'rp', 'rd', 'ori', 'full']:
            accs = []
            img_root = os.path.join(data_dir, kind)
            dataset_ine = ImageNet_E(img_root, labels, transform=ine_transform)
            sampler = None
            ine_data_loader = torch.utils.data.DataLoader(
                dataset_ine, sampler=sampler,
                batch_size=args.batch_size, shuffle=False)
            acc = validate(ine_data_loader, model, logger)
            logger.info(f"Accuracy: {kind}: {acc} ")





