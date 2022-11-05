import cv2
import torch
import os
import numpy as np
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        super(CustomDataset, self).__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        label = row['liveness_score']
        uid = row['uid']
        img = cv2.imread(img_path)
        assert img is not None, img_path

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        label = torch.as_tensor([label])
        return {
            'image': img,
            'uid': uid,
            'fname': os.path.basename(img_path),
            'label': label
        }

class InferDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        super(InferDataset, self).__init__()
        self.img_files = glob.glob(os.path.join(img_dir, '**', '*'), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        assert img is not None, img_path

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img)
        return {
            'image': img,
            'fname': os.path.basename(img_path)
        }