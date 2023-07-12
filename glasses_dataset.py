import os
import sklearn
import PIL.Image
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, is_train=True):
        file = "data/train.csv"
        prefix = "Train"
        self.img_dir = "data/images"
        self.images = [filename for filename in os.listdir(self.img_dir) if filename.startswith(prefix)]
        train, test = sklearn.model_selection.train_test_split(self.images,train_size=0.8)
        if is_train:
            self.images = train
        else:
            self.images = test

        self.img_labels = pd.read_csv(file)

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_with_ext = self.images[idx]
        img_path = os.path.join(self.img_dir, image_with_ext)
        image = PIL.Image.open(img_path)
        image = self.transforms(image)

        name = image_with_ext[0:-4]
        label = self.img_labels[self.img_labels["image_id"] == name].iloc[0]["healthy"]
        return image, torch.tensor(label)


if __name__ == "__main__":
    cid = CustomImageDataset()
    dataloader = DataLoader(cid, batch_size=50, shuffle=True)
    
    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
        print(image[0])
        exit(0)
