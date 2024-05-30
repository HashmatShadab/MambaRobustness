import json
import os
import argparse
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.utils as vutils
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from dataset import ImageNet5k
import torchvision.models as models

from timm.models import create_model
import timm
# import vit_models_ipvit as vit_models
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
try:
    from inference import load_pretrained_ema, load_mamba_models
except ImportError as e:
    print(f"Error importing: {e}")

def plot_grid(w, save=False, name="grid.png"):
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    if save:
        plt.savefig(name)
    plt.show()

    
def get_model(model_name=None):
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
    
    elif  'dino_small' in model_name:
        import vit_models_ipvit as vit_models
        model = vit_models.dino_small(patch_size=16, pretrained=True)

    if model_name not in ['vssm_tiny_v0', 'vssm_small_v0', 'dino_small', 'vssm_base_v0', 'vssm_tiny_v2', 'vssm_small_v2', 'vssm_base_v2']:
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if "dino_small" not in model_name:

        model = nn.Sequential(Normalize(mean, std), model)


    return model, mean, std


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


def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t
def parse_args():

    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--test_dir', default=r'F:\Code\datasets\ImageNet\val',
                        help='ImageNet Validation Data')
    parser.add_argument('--exp_name', default=None, help='pretrained weight path')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model Name')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')

    parser.add_argument('--drop_count', type=int, default=180, help='How many patches to drop')
    parser.add_argument('--drop_best', action='store_true', default=False, help="set True to drop the best matching")


    parser.add_argument('--test_image', action='store_true', default=False, help="set True to output test images")

    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle instead of dropping")
    parser.add_argument('--shuffle_size', type=int, default=2, help='nxn grid size of n', nargs='*')
    parser.add_argument('--shuffle_h', type=int, default=None, help='h of hxw grid', nargs='*')
    parser.add_argument('--shuffle_w', type=int, default=None, help='w of hxw grid', nargs='*')
    parser.add_argument('--random_drop', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--random_drop_v2', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--random_drop_v2_increase_forward', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--random_drop_v2_max_at_center', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--random_drop_v2_min_at_center', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--random_drop_v2_increase_backward', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--v2_direction', type=str, default="1", choices=["1", "2", "3", "4"] )
    parser.add_argument('--random_drop_v3', action='store_true', default=False, help="randomly drop patches")






    parser.add_argument('--random_offset_drop', action='store_true', default=False, help="randomly drop patches")

    parser.add_argument('--cascade', action='store_true', default=False, help="run cascade evaluation")
    parser.add_argument('--exp_count', type=int, default=2, help='random experiment count to average over')
    parser.add_argument('--saliency', action='store_true', default=False, help="drop using saliency")
    parser.add_argument('--saliency_box', action='store_true', default=False, help="drop using saliency")

    parser.add_argument('--drop_lambda', type=float, default=0.2, help='percentage of image to drop for box')
    parser.add_argument('--standard_box', action='store_true', default=False, help="drop using standard model")
    parser.add_argument('--dino', action='store_true', default=False, help="drop using dino model saliency")

    parser.add_argument('--lesion', action='store_true', default=False, help="drop using dino model saliency")
    parser.add_argument('--block_index', type=int, default=0, help='block index for lesion method', nargs='*')

    parser.add_argument('--draw_plots', action='store_true', default=False, help="draw plots")
    parser.add_argument('--select_im', action='store_true', default=False, help="select robust images")
    parser.add_argument('--save_path', type=str, default=None, help='save path')

    # segmentation evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.9, help='threshold for segmentation')
    parser.add_argument('--pretrained_weights', default=None, help='pretrained weights path')
    parser.add_argument('--patch_size', type=int, default=16, help='nxn grid size of n')
    parser.add_argument('--use_shape', action='store_true', default=False, help="use shape token for prediction")
    parser.add_argument('--rand_init', action='store_true', default=False, help="use randomly initialized model")
    parser.add_argument('--generate_images', action='store_true', default=False, help="generate images instead of eval")

    return parser.parse_args()



def main(args, device, verbose=True):
    if verbose:
        if args.shuffle:
            print(f"Shuffling inputs and evaluating {args.model_name}")
        elif args.random_drop:
            print(f"{args.model_name} dropping {args.drop_count} random patches")


        elif args.random_drop_v2:
            print(f"{args.model_name} dropping {args.drop_count * 100} % random pixels from each patch")
        elif args.random_drop_v3:
            print(f"{args.model_name} dropping {args.drop_count * 100} % patches sequentially from the start with direction {args.v2_direction}")
        elif args.random_drop_v2_increase_forward:
            print(f"{args.model_name} dropping {args.drop_count * 100} % random pixels in increasing order with direction {args.v2_direction}")
        elif args.random_drop_v2_increase_backward:
            print(f"{args.model_name} dropping {args.drop_count * 100} % random pixels in decreasing order with direction {args.v2_direction}")
        elif args.random_drop_v2_max_at_center:
            print(f"{args.model_name} dropping {args.drop_count * 100} % random pixels with maximum at the center with direction {args.v2_direction}")
        elif args.random_drop_v2_min_at_center:
            print(f"{args.model_name} dropping {args.drop_count * 100} % random pixels with minimum at the center with direction {args.v2_direction}")


        elif args.lesion:
            print(f"{args.model_name} dropping {args.drop_count} random patches from block {args.block_index}")
        elif args.cascade:
            print(f"evaluating {args.model_name} in cascade mode")
        elif args.saliency:
            print(f"{args.model_name} dropping {'most' if args.drop_best else 'least'} "
                  f"salient {args.drop_count} patches")
        elif args.saliency_box:
            print(f"{args.model_name} dropping {args.drop_lambda} % most salient pixels")
        elif args.standard_box:
            print(f"{args.model_name} dropping {args.drop_lambda} % pixels around most matching patch")
        elif args.dino:
            print(f"{args.model_name} picking {args.drop_lambda * 100} %  "
                  f"{'foreground' if args.drop_best else 'background'} pixels using dino")
        else:
            print(f"{args.model_name} dropping {'least' if args.drop_best else 'most'} "
                  f"matching {args.drop_count} patches")

    if args.dino:
        cur_model_name = args.model_name
        args.model_name = "dino_small"
        dino_model, _, _ = get_model(args.model_name)
        args.model_name = cur_model_name
        dino_model.to(device)
        dino_model.eval()

    model, mean, std = get_model(model_name=args.model_name)

    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith("https://"):
            ckpt = torch.hub.load_state_dict_from_url(url=args.pretrained_weights, map_location="cpu")
        else:
            ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        if "model" in ckpt:
            msg = model.load_state_dict(ckpt["model"])
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                name = k[7:]  # remove `module.' from state dict
                new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict)

        print(msg)
    model = model.to(device)
    model.eval()

    # print model parameters
    if verbose:
        print(f"Parameters in Millions: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.3f}")

    # Setup-Data
    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    # Test Samples
    test_dir = args.test_dir
    test_set = ImageNet5k(root=test_dir, transform=data_transform)
    test_size = len(test_set)
    if verbose:
        print(f'Test data size: {test_size}')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    similarity_measure = torch.nn.CosineSimilarity(dim=2, eps=1e-08)

    clean_acc = 0.0
    for i, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            img, label = img.to(device), label.to(device)

            if args.shuffle or args.random_drop or args.random_drop_v2 or args.random_drop_v3 or args.random_drop_v2_increase_forward or args.random_drop_v2_max_at_center or args.random_drop_v2_min_at_center or args.random_drop_v2_increase_backward:
                if isinstance(args.shuffle_size, int):
                    assert 224 % args.shuffle_size == 0, f"shuffle size {args.shuffle_size} " \
                                                         f"not compatible with 224 image"
                    shuffle_h, shuffle_w = args.shuffle_size, args.shuffle_size
                    patch_dim1, patch_dim2 = 224 // args.shuffle_size, 224 // args.shuffle_size
                    patch_num = args.shuffle_size * args.shuffle_size
                else:
                    shuffle_h, shuffle_w = args.shuffle_size
                    patch_dim1, patch_dim2 = 224 // shuffle_h, 224 // shuffle_w
                    patch_num = shuffle_h * shuffle_w

                # patch_num gives the total number of patches in the image
                # patch_dim1 and patch_dim2 give the dimensions of each patch
                # shuffle_h and shuffle_w give the number of patches in each dimension

                if args.random_drop_v2 or args.random_drop_v3 or args.random_drop_v2_increase_forward or args.random_drop_v2_max_at_center or args.random_drop_v2_min_at_center or args.random_drop_v2_increase_backward:
                    # get total number of pixels within each patch
                    pixels_per_patch = patch_dim1 * patch_dim2

                if args.random_offset_drop:
                    mask = torch.ones_like(img)
                    mask = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)


                if args.random_drop_v2 or args.random_drop_v3 or args.random_drop_v2_increase_forward or args.random_drop_v2_max_at_center or args.random_drop_v2_min_at_center or args.random_drop_v2_increase_backward:

                    img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_dim1, p2=patch_dim2)
                    if args.v2_direction == "1":
                        # print("Direction 1: No change in direction")
                        pass
                    elif args.v2_direction == "2":
                        # print("Direction 2: Transpose the image")
                        # flip the direction of the patches
                        img = rearrange(img, 'b (h w) (p1 p2) c -> b (w h) (p2 p1) c', h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                    elif args.v2_direction == "3":
                        # print("Direction 3: Transpose the image and reverse the order of the patches")
                        # direction 2
                        img = rearrange(img, 'b (h w) (p1 p2) c -> b (w h) (p2 p1) c', h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                        # reverse the order of the patches
                        img = torch.flip(img, dims=[1])
                    elif args.v2_direction == "4":
                        # print("Direction 4: Reverse the order of the patches")
                        # flip the direction of the patches
                        img = torch.flip(img, dims=[1])
                else:
                    # print("No change in direction")
                    img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)


                if args.shuffle:
                    row = np.random.choice(range(patch_num), size=img.shape[1], replace=False)
                    img = img[:, row, :]  # images have been shuffled already
                elif args.random_drop and args.drop_count > 0:
                    row = np.random.choice(range(patch_num), size=args.drop_count, replace=False)
                    if args.random_offset_drop:
                        mask[:, row, :] = 0.0
                    else:
                        img[:, row, :] = 0.0



                elif args.random_drop_v3 and args.drop_count > 0 :
                    # Drop number of patches equal to the percentage given by drop_count
                    total_drop_count = int(patch_num * args.drop_count)
                    # pacthes will be dropped in the forward direction
                    row = np.arange(total_drop_count)
                    img[:, row, :] = 0.0

                elif args.random_drop_v2 and args.drop_count > 0 :
                    # for each patch, drop random number of pixels equal to the percentage given by drop_count

                    for patch_idx in range(img.shape[1]):
                        drop_count = int(pixels_per_patch * args.drop_count)
                        # print(f"patch_idx: {patch_idx}, drop_count: {drop_count}")
                        row = np.random.choice(range(pixels_per_patch), size=drop_count, replace=False)
                        img[:, patch_idx, row] = 0.0

                elif args.random_drop_v2_increase_forward and args.drop_count > 0 :
                    # int(pixels_per_patch * args.drop_count) is the maximum number of pixels to drop from a patch
                    # linearly increase the number of pixels to drop from each patch in the forward direction

                    max_drop_count = int(pixels_per_patch * args.drop_count)

                    for patch_idx in range(img.shape[1]):
                        drop_count = int(pixels_per_patch * args.drop_count * (patch_idx+1) / img.shape[1])
                        # print(f"patch_idx: {patch_idx}, drop_count: {drop_count}")
                        row = np.random.choice(range(pixels_per_patch), size=drop_count, replace=False)
                        img[:, patch_idx, row] = 0.0

                elif args.random_drop_v2_increase_backward and args.drop_count > 0 :
                    # int(pixels_per_patch * args.drop_count) is the maximum number of pixels to drop from a patch
                    # linearly decrease the number of pixels to drop from each patch in the forward direction

                    max_drop_count = int(pixels_per_patch * args.drop_count)

                    for patch_idx in range(img.shape[1]):
                        drop_count = int(pixels_per_patch * args.drop_count * (patch_idx+1) / img.shape[1])
                        drop_count = max_drop_count - drop_count
                        # print(f"patch_idx: {patch_idx}, drop_count: {drop_count}")
                        row = np.random.choice(range(pixels_per_patch), size=drop_count, replace=False)
                        img[:, patch_idx, row] = 0.0

                elif args.random_drop_v2_max_at_center and args.drop_count > 0 :
                    # int(pixels_per_patch * args.drop_count) is the maximum number of pixels to drop from a patch
                    # linearly increase the number of pixels till the center of the image and then decrease

                    max_drop_count = int(pixels_per_patch * args.drop_count)

                    for patch_idx in range(img.shape[1]):
                        if patch_idx < img.shape[1] // 2:
                            drop_count = int(pixels_per_patch * args.drop_count * (patch_idx+1) / (img.shape[1] // 2))
                        else:
                            drop_count = int(pixels_per_patch * args.drop_count * (img.shape[1] - patch_idx) / (img.shape[1] // 2))
                        # print(f"patch_idx: {patch_idx}, drop_count: {drop_count}")
                        row = np.random.choice(range(pixels_per_patch), size=drop_count, replace=False)
                        img[:, patch_idx, row] = 0.0

                elif args.random_drop_v2_min_at_center and args.drop_count > 0 :
                    # int(pixels_per_patch * args.drop_count) is the maximum number of pixels to drop from a patch
                    # linearly decrease the number of pixels  to be dropped till the center of the image and then increase

                    max_drop_count = int(pixels_per_patch * args.drop_count)

                    for patch_idx in range(img.shape[1]):
                        if patch_idx < img.shape[1] // 2:
                            drop_count = int(pixels_per_patch * args.drop_count * (patch_idx + 1) / (img.shape[1] // 2))
                        else:
                            drop_count = int(pixels_per_patch * args.drop_count * (img.shape[1] - patch_idx) / (img.shape[1] // 2))

                        drop_count = max_drop_count - drop_count
                        # print(f"patch_idx: {patch_idx}, drop_count: {drop_count}")
                        row = np.random.choice(range(pixels_per_patch), size=drop_count, replace=False)
                        img[:, patch_idx, row] = 0.0





                if args.random_drop_v2 or args.random_drop_v2_increase_forward or args.random_drop_v2_max_at_center or args.random_drop_v2_min_at_center or args.random_drop_v2_increase_backward or args.random_drop_v3:
                    if args.v2_direction == "1":
                        # print("Direction 1")
                        pass
                    elif args.v2_direction == "2":
                        # print("Direction 2")
                        img = rearrange(img, 'b (w h) (p2 p1) c -> b (h w) (p1 p2) c', h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                    elif args.v2_direction == "3":
                        # print("Direction 3")
                        img = torch.flip(img, dims=[1])
                        img = rearrange(img, 'b (w h) (p2 p1) c -> b (h w) (p1 p2) c', h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                    elif args.v2_direction == "4":
                        # print("Direction 4")
                        img = torch.flip(img, dims=[1])


                    img = rearrange(img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                    h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)

                    # img = rearrange(img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                    #                 h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                else:

                    img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)

                if args.random_offset_drop and args.drop_count > 0:
                    mask = rearrange(mask, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                    new_mask = torch.ones_like(mask)
                    mask_off_set = 8
                    new_mask[:, :, mask_off_set:, mask_off_set:] = mask[:, :, :-mask_off_set, :-mask_off_set]
                    img = new_mask * img

            elif args.dino:
                head_number = 1

                # attentions = dino_model.forward_selfattention(normalize(img.clone(), mean=mean, std=std))
                attentions = dino_model.forward_selfattention(img)
                attentions = attentions[:, head_number, 0, 1:]

                w_featmap = int(np.sqrt(attentions.shape[-1]))
                h_featmap = int(np.sqrt(attentions.shape[-1]))
                scale = img.shape[2] // w_featmap

                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.drop_lambda)
                idx2 = torch.argsort(idx)
                for batch_idx in range(th_attn.shape[0]):
                    th_attn[batch_idx] = th_attn[batch_idx][idx2[batch_idx]]

                th_attn = th_attn.reshape(-1, w_featmap, h_featmap).float()
                th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(1), scale_factor=scale, mode="nearest")

                if args.drop_best:  # foreground
                    img = img * (1 - th_attn)
                else:
                    img = img * th_attn

            else:
                pass

            if args.test_image:
                if args.shuffle:
                    if isinstance(args.shuffle_size, int):
                        save_name = args.shuffle_size
                    else:
                        save_name = f"{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    save_path = f"non-adversarial_logs/shuffle_results/images"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{save_name}.jpg")
                elif args.random_drop:
                    save_path = f"report/random/images"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.random_drop_v2:
                    save_path = f"report/random_v2_{args.v2_direction}/images_{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")

                elif args.random_drop_v3:
                    save_path = f"report/random_v3_{args.v2_direction}/images_{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.random_drop_v2_increase_forward:
                    save_path = f"report/random_v2_increase_forward_{args.v2_direction}/images_{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.random_drop_v2_increase_backward:
                    save_path = f"report/random_v2_increase_backward_{args.v2_direction}/images_{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")

                elif args.random_drop_v2_max_at_center:
                    save_path = f"report/random_v2_max_at_center_{args.v2_direction}/images_{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.random_drop_v2_min_at_center:
                    save_path = f"report/random_v2_min_at_center_{args.v2_direction}/images_{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.dino:
                    save_path = f"report/dino/images"
                    drop_order = 'foreground' if args.drop_best else 'background'
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/image_{drop_order}_{args.drop_lambda}.jpg")
                else:
                    pass
                return 0

            if args.lesion:
                if "resnet" in args.model_name:
                    clean_out = model(img.clone(), drop_layer=args.block_index,
                                      drop_percent=args.drop_count)
                else:
                    clean_out = model(img.clone(), block_index=args.block_index,
                                      drop_rate=args.drop_count)
            else:
                clean_out = model(img.clone())

            if isinstance(clean_out, list):
                clean_out = clean_out[-1]
            clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()

    print(f"{args.model_name} Top-1 Accuracy: {clean_acc / len(test_set)}")
    return clean_acc / len(test_set)


if __name__ == '__main__':
    opt = parse_args()

    acc_dict = {}

    if opt.shuffle:
        if opt.shuffle_h is not None:
            assert opt.shuffle_w is not None, "need to specify both shuffle_h and shuffle_w!"
            assert len(opt.shuffle_h) == len(opt.shuffle_w), "mismatch for shuffle h, w pairs"
            shuffle_list = list(zip(opt.shuffle_h, opt.shuffle_w))
        else:
            shuffle_list = opt.shuffle_size
        if isinstance(shuffle_list, int):
            shuffle_list = [shuffle_list, ]
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for shuffle_size in shuffle_list:
                opt.shuffle_size = shuffle_size
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                if isinstance(shuffle_size, tuple):
                    shuffle_size = shuffle_size[0] * shuffle_size[1]
                acc_dict[f"run_{rand_exp:03d}"][f"{shuffle_size}"] = acc
        if not opt.test_image:
            json.dump(acc_dict, open(f"report_shuffle_{opt.model_name}.json", "w"), indent=4)

    elif opt.random_drop:
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(1, 10):
                if isinstance(opt.shuffle_size, list):
                    opt.drop_count = drop_count * opt.shuffle_size[0] * opt.shuffle_size[1] // 10
                else:
                    opt.drop_count = drop_count * 196 // 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_{opt.model_name}{shuffle_name}.json", "w"), indent=4)


    elif opt.random_drop_v2:

        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 11):

                opt.drop_count = drop_count / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_v2_{opt.v2_direction}_{opt.model_name}{shuffle_name}.json", "w"), indent=4)

    elif opt.random_drop_v3:

        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 11):
                opt.drop_count = drop_count / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_v3_{opt.v2_direction}_{opt.model_name}{shuffle_name}.json", "w"), indent=4)


    elif opt.random_drop_v2_increase_forward:

        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 11):

                opt.drop_count = drop_count / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_v2_increase_forward_{opt.v2_direction}_{opt.model_name}{shuffle_name}.json", "w"), indent=4)

    elif opt.random_drop_v2_increase_backward:

        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 11):

                opt.drop_count = drop_count / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_v2_increase_backward_{opt.v2_direction}_{opt.model_name}{shuffle_name}.json", "w"), indent=4)

    elif opt.random_drop_v2_max_at_center:

        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 11):

                opt.drop_count = drop_count / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_v2_max_at_centre_{opt.v2_direction}_{opt.model_name}{shuffle_name}.json", "w"), indent=4)

    elif opt.random_drop_v2_min_at_center:

        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 11):

                opt.drop_count = drop_count / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc

        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report_random_v2_min_at_centre_{opt.v2_direction}_{opt.model_name}{shuffle_name}.json", "w"), indent=4)



    elif opt.dino:
        for drop_best in [True, False]:
            opt.drop_best = drop_best
            acc_dict[f"{'best' if opt.drop_best else 'worst'}"] = {}
            for drop_lambda in range(1, 11):
                opt.drop_lambda = drop_lambda / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"{'best' if opt.drop_best else 'worst'}"][f"{drop_lambda}"] = acc
        if not opt.test_image:
            json.dump(acc_dict, open(f"report_dino_{opt.model_name}.json", "w"), indent=4)

    elif opt.lesion:
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            block_index_list = opt.block_index
            for cur_block_num in block_index_list:
                opt.block_index = cur_block_num
                acc_dict[f"run_{rand_exp:03d}"][f"{cur_block_num}"] = {}
                for drop_count in [0.25, 0.50, 0.75]:
                    opt.drop_count = drop_count
                    acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    acc_dict[f"run_{rand_exp:03d}"][f"{cur_block_num}"][f"{drop_count}"] = acc
        if not opt.test_image:
            json.dump(acc_dict, open(f"report/lesion/{opt.model_name}.json", "w"), indent=4)

    else:
        print("No arguments specified: finished running")