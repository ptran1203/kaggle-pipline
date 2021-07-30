import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, df, transform=None, img_size=512, return_path=False):
        self.df = df
        self.transform = transform
        self.img_size = img_size
        self.return_path = return_path

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

        image = torch.tensor(image)
        # Channel last -> channel first
        image = image.permute(2, 0, 1)

        if self.return_path:
            return path, image, target
        else:
            return image, target
