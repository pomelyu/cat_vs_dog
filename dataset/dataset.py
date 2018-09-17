import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class DogCatData(Dataset):
    def __init__(self, dir_path, usage="train", valid_ratio=0.2):
        self.usage = usage

        imgs  = [os.path.join(dir_path, name) for name in os.listdir(dir_path) if name.endswith(".jpg")]
        imgs = sorted(imgs, key=lambda x: int(x.split("/")[-1].split(".")[-2]))
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if usage == "test":
            self.data = imgs
        elif usage == "train":
            index = int(len(imgs) * (1-valid_ratio))
            self.data = imgs[:index]
        elif usage == "valid":
            index = int(len(imgs) * (1-valid_ratio))
            self.data = imgs[index:]
        else:
            print("Error: usage must be one of 'test', 'train' or 'valid'")

        if usage == "train":
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ])

    def __getitem__(self, index):
        image_path = self.data[index]

        if self.usage == "test":
            label = int(image_path.split("/")[-1].split(".")[0])
        else:
            class_name = image_path.split("/")[-1].split(".")[0]
            label = 1 if class_name.find("dog") != -1 else 0

        data = Image.open(image_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.data)
