import torch
import torch.nn as nn
import argparse
import timm

try:
    from inference import load_pretrained_ema, load_mamba_models
except ImportError as e:
    print(f"Error importing: {e}")

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageNet5k
import os
from attacks import PGD, FGSM, BIM, MIFGSM, DIFGSM, TPGD, TIFGSM, VMIFGSM
import json
import torch_dct as dct
import matplotlib.pyplot as plt
import torchvision


def plot_grid(w, save=False, name="grid.png"):
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    if save:
        plt.savefig(name)
    plt.show()


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


def get_val_loader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # Load ImageNet validation dataset
    val_dataset = ImageNet5k(root=os.path.join(data_path, "val"), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader, val_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Transferability test')
    parser.add_argument('--data_dir', help='path to ImageNet dataset', default=r'F:\Code\datasets\ImageNet')
    parser.add_argument('--attack_name', default='fgsm')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--attack_steps', default=20, type=int)
    parser.add_argument('--filter', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--filter_size', type=int, default=32)
    parser.add_argument('--filter_preserve', default='low', type=str, choices=['low', 'high'])
    parser.add_argument('--source_model_name', default='resnet18')
    parser.add_argument('--epsilon', type=int, default=2)
    parser.add_argument('--save_results_only', type=lambda x: (str(x).lower() == 'true'), default=True)

    args = parser.parse_args()

    return args


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


def evaluate_models(original_images, adversarial_images, model1, model2, labels, num_classes=1000):
    correct_1_adv = 0
    correct_2_adv = 0
    correct_1 = 0
    correct_2 = 0

    confusion_matrix_1 = torch.zeros(num_classes, num_classes)
    confusion_matrix_1_adv = torch.zeros(num_classes, num_classes)
    confusion_matrix_2 = torch.zeros(num_classes, num_classes)
    confusion_matrix_2_adv = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        outputs_1_adv = model1(adversarial_images)
        outputs_2_adv = model2(adversarial_images)
        outputs_1 = model1(original_images)
        outputs_2 = model2(original_images)

        _, predicted_1 = outputs_1.max(1)
        _, predicted_2 = outputs_2.max(1)
        _, predicted_1_adv = outputs_1_adv.max(1)
        _, predicted_2_adv = outputs_2_adv.max(1)

        correct_1 += torch.sum(predicted_1 == labels)
        correct_2 += torch.sum(predicted_2 == labels)
        correct_1_adv += torch.sum(predicted_1_adv == labels)
        correct_2_adv += torch.sum(predicted_2_adv == labels)

        for t, p in zip(labels.view(-1), predicted_1.view(-1)):
            confusion_matrix_1[t.long(), p.long()] += 1
        for t, p in zip(labels.view(-1), predicted_1_adv.view(-1)):
            confusion_matrix_1_adv[t.long(), p.long()] += 1

        for t, p in zip(labels.view(-1), predicted_2.view(-1)):
            confusion_matrix_2[t.long(), p.long()] += 1
        for t, p in zip(labels.view(-1), predicted_2_adv.view(-1)):
            confusion_matrix_2_adv[t.long(), p.long()] += 1

    return correct_1, correct_1_adv, correct_2, correct_2_adv, confusion_matrix_1, confusion_matrix_2, confusion_matrix_1_adv, confusion_matrix_2_adv


def transfer_attack(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filter_size = args.filter_size



    epsilon_list = [args.epsilon]

    val_loader, val_dataset = get_val_loader(data_path=args.data_dir, batch_size=args.batch_size)

    steps = [1] if args.attack_name == 'fgsm' else [args.attack_steps]


    source_models = [args.source_model_name]

    for step in steps:

        for eps in epsilon_list:

            for source_model_name in source_models:

                print(f"Generating adversarial images for {source_model_name} using {args.attack_name} with epsilon {eps} and step {step}")
                source_model, mean1, std1 = get_model(source_model_name, device=device)
                source_model = source_model.eval()
                if args.attack_name == 'pgd':
                    if args.filter:
                        attack = PGD(source_model, eps=eps / 255, alpha=2 / 255, steps=step, random_start=True, return_delta=True)
                    else:
                        attack = PGD(source_model, eps=eps / 255, alpha=2 / 255, steps=step, random_start=True)
                elif args.attack_name == 'fgsm':
                    attack = FGSM(source_model, eps=eps / 255)
                elif args.attack_name == 'bim':
                    attack = BIM(source_model, eps=eps / 255, alpha=2 / 255, steps=step)
                elif args.attack_name == 'mifgsm':
                    attack = MIFGSM(source_model, eps=eps / 255, alpha=2 / 255, steps=step, decay=1.0)
                elif args.attack_name == 'difgsm':
                    attack = DIFGSM(source_model, eps=eps / 255, alpha=2 / 255, steps=step, decay=0.0,
                                    resize_rate=0.9, diversity_prob=0.5, random_start=False)
                elif args.attack_name == 'tpgd':
                    attack = TPGD(source_model, eps=eps / 255, alpha=2 / 255, steps=step)
                elif args.attack_name == 'tifgsm':
                    attack = TIFGSM(source_model, eps=eps / 255, alpha=2 / 255, steps=step, decay=0.0, resize_rate=0.9,
                                    diversity_prob=0.5, random_start=False, kernel_name='gaussian', len_kernel=15, nsig=3)
                elif args.attack_name == 'vmifgsm':
                    attack = VMIFGSM(source_model, eps=eps / 255, alpha=2 / 255, steps=step, decay=1.0, N=5, beta=3 / 2)


                if args.filter:
                    data_path = f"AdvExamples_freq/{args.attack_name}_eps_{eps}_steps_{step}_{args.filter_preserve}_{args.filter_size}/{source_model_name}"
                    if args.save_results_only:
                        data_path = f"AdvExamples_freq_results/{source_model_name}/{args.attack_name}_eps_{eps}_steps_{step}_{args.filter_preserve}_{args.filter_size}"

                else:
                    data_path = f"AdvExamples/{args.attack_name}_eps_{eps}_steps_{step}/{source_model_name}"
                    if args.save_results_only:
                        data_path = f"AdvExamples_results/{source_model_name}/{args.attack_name}_eps_{eps}_steps_{step}"

                if not os.path.exists(data_path):
                    os.makedirs(data_path, exist_ok=True)


                images_list = []
                labels_list = []
                correct = 0
                total = 0
                correct_adv = 0

                for idx, (images, labels) in enumerate(val_loader):

                    images, labels = images.to(device), labels.to(device)

                    if args.attack_name == 'pgd' and args.filter:
                        adversarial_images, delta = attack(images, labels)
                        grad = delta.clone().detach_()
                        freq = dct.dct_2d(grad)
                        if args.filter_preserve == 'low':
                            mask = torch.zeros(freq.size()).to(device)
                            mask[:, :, :filter_size, :filter_size] = 1
                        elif args.filter_preserve == 'high':
                            mask = torch.zeros(freq.size()).to(device)
                            mask[:, :, filter_size:, filter_size:] = 1
                        masked_freq = torch.mul(freq, mask)
                        new_grad = dct.idct_2d(masked_freq)
                        adversarial_images = torch.clamp(images + new_grad, 0, 1).detach_()

                    else:
                        adversarial_images = attack(images, labels)


                    outputs = source_model(adversarial_images)
                    _, predicted = outputs.max(1)
                    correct_adv += predicted.eq(labels).sum().item()

                    outputs = source_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()

                    total += labels.size(0)

                    if not args.save_results_only:
                        images_list.append(adversarial_images.detach().cpu())
                        labels_list.append(labels.detach().cpu())

                    print(f"Batch {idx + 1} / {len(val_loader)}")

                if not args.save_results_only:
                    images = torch.cat(images_list, dim=0)
                    labels = torch.cat(labels_list, dim=0)

                print(f"Accuracy of the network on the {total} test images: {100 * correct / total}%")
                print(f"Accuracy of the network on the {total} adversarial images: {100 * correct_adv / total}%")

                # save the adversarial images and labels together so that later we can load them using TensorDataset

                # create a txt file in the same directory to store the accuracy
                if args.save_results_only:
                    with open(f"{data_path}/accuracy.txt", 'w') as file:
                        file.write(f"Accuracy of the network on the {total} test images: {100 * correct / total}%\n")
                        file.write(f"Accuracy of the network on the {total} adversarial images: {100 * correct_adv / total}%\n")
                else:
                    torch.save((images, labels), f"{data_path}/images_labels.pt")







if __name__ == "__main__":
    args = get_args()
    transfer_attack(args)

