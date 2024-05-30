import torchvision
import json
import torch
import os
class ImageNet5k(torchvision.datasets.ImageFolder):

    def __init__(self, image_list="./image_list.json", *args, **kwargs):
        self.image_list = set(json.load(open(image_list, "r"))["images"])
        super(ImageNet5k, self).__init__(is_valid_file=self.is_valid_file, *args, **kwargs)

    def is_valid_file(self, x: str) -> bool:

        file_path = x
        # get image name
        image_name = os.path.basename(file_path)
        # get parent folder name
        folder_name = os.path.basename(os.path.dirname(file_path))

        return f"{folder_name}/{image_name}" in self.image_list



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import os

    # Load the image list


    # Load the ImageNet dataset
    imagenet = ImageNet5k(root=r"F:\Code\datasets\ImageNet\val", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))

    dataloader = torch.utils.data.DataLoader(imagenet, batch_size=50, shuffle=True)

    for i, (img, label) in enumerate(dataloader):
        print(i, img.shape)
