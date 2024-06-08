from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
from PIL import Image
import json
import torchvision
import torch
import shutil
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class AdeData(torchvision.datasets.VisionDataset):

    def __init__(self, ann_path, root, transform=None, corruption=None, severity=None,
                 save_path=None, loader=default_loader):
        self.root = root
        self.ann_path = ann_path
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.save_path = save_path
        self.loader = loader

        # get all the filenames of the images in the root directory
        self.data_list = os.listdir(self.root)


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index: int):

        filename = self.data_list[index]
        image_path = os.path.join(self.root, filename)

        image = self.loader(image_path)

        if self.transform:
            # image = self.transform(image)
            # get numpy array from image
            image = np.array(image)
            image = corrupt(image, corruption_name=self.corruption, severity=self.severity)

        save_path = os.path.join(self.save_path, self.corruption, str(self.severity), 'images', 'validation')
        corrupt_annotation_path = os.path.join(self.save_path, self.corruption, str(self.severity), 'annotations', 'validation')

        if not os.path.exists(corrupt_annotation_path):
            os.makedirs(corrupt_annotation_path, exist_ok=True)
            # loop over the annotation files and copy them to the new directory
            for ann_file in os.listdir(self.ann_path):
                shutil.copy(os.path.join(self.ann_path, ann_file), corrupt_annotation_path)




            # shutil.copy(self.ann_path, corrupt_annotation_path)



        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)



        image_name = filename

        image_path = os.path.join(save_path, image_name)
        image = Image.fromarray(image)
        image.save(image_path)

        return 0


def get_args():
    parser = argparse.ArgumentParser()



    # Task Data Parameters

    parser.add_argument("--data_path", default=r"data/ade/ADEChallengeData2016", type=str, help='path of the clean images')

    # Logging Parameters
    parser.add_argument("--save_path", default=r"aed_corrupted", type=str, help='the folder name of output')


    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    images_path = os.path.join(args.data_path,"images" ,  "validation")
    annotations_path = os.path.join(args.data_path, "annotations", "validation")

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),])

    for corruption in get_corruption_names():
        tic = time.time()
        for severity in range(5):
            dataset = AdeData(ann_path = annotations_path,root=images_path, transform=transforms,
                                         corruption=corruption, severity=severity+1, save_path=args.save_path)
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

            for i, (image) in enumerate(data_loader):
                print(i)
        print(corruption, time.time() - tic)



