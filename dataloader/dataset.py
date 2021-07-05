import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from dataloader.augment import train_transform, valid_transform


class MyDataset(Dataset):
    def __init__(self, df, is_train=True, img_size=512):
        self.df = df
        self.transform = train_transform() if is_train else valid_transform()
        self.img_size = img_size

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']

        image = cv2.imread(path)[:,:,::-1].astype(np.uint8)
        image = cv2.resize(image, (self.img_size, self.img_size))

        target = self.df.loc[idx, 'label']
        target = torch.tensor(target)

        if self.transform:
            image = self.transform(image=image)['image']

        # Channel last -> channel first
        image = image.permute(2, 0, 1)

        return image, target
