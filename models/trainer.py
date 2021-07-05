import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.dataset import MyDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.criterion = LossFunction()

        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, train_loader, e):
        losses = AverageMeter()
        self.model.train()

        loop = tqdm(train_loader)

        for image, labels in loop:
            self.optimizer.zero_grad()

            image = image.to(self.device)
            labels= labels.to(self.device)

            output = self.model(image)
            loss = self.criterion(output, labels)
            loss.backward()

            loss = loss.cpu().detach().numpy()
            losses.update(loss, self.args.batch_size)

            self.optimizer.step()

            loop.set_description(f"Epoch {e + 1}")
            loop.set_postfix(loss=loss)

            
        return losses.avg

    def val_one_epoch(self, loader):
        losses = AverageMeter()

        self.model.eval()
        
        for image,labels in loader:
            image = image.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, labels)
            loss = loss.cpu().detach().numpy()
            losses.update(loss, self.args.batch_size)
            
        return losses.avg

    def get_loaders(self, df, fold):
        train_data = df[df.fold != fold].reset_index(drop=True)
        val_data = df[df.fold == fold].reset_index(drop=True)

        train_data = MyDataset(train_data, transform=train_transform(),
                               img_size=self.args.img_size)

        val_data= MyDataset(val_data, transform=val_transform(),
                            img_size=self.args.img_size)
        
        train_loader = DataLoader(train_data,
                                shuffle=True,
                                num_workers=0,
                                batch_size=self.args.batch_size)

        valid_loader = DataLoader(val_data,
                                shuffle=False,
                                num_workers=0,
                                batch_size=self.args.batch_size)
        
        return train_loader, valid_loader

    def save_checkpoint(self, fold):
        path = f'{self.model.name}_fold{fold}.pth'
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, fold):
        pass

    def train(self, train_df, fold, epochs=20):
        if self.args.resume:
            self.load_checkpoint(fold)

        train_loader, valid_loader = self.get_loaders(train_df, fold)

        print(f'Train on {len(train_loader)} images, validate on {len(valid_loader)} images')

        steps_per_epoch = len(train_loader) // self.args.batch_size

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, verbose=True, T_max=epochs * steps_per_epoch
        )

        best_loss = 1000.0
        for e in range(epochs):
            train_loss = self.train_one_epoch(train_loader, e)

            scheduler.step()

            val_loss = self.val_one_epoch(valid_loader)

            if val_loss < best_loss:
                print(f'Valid loss improved from {best_loss} to {val_loss}')
                best_loss = val_loss
                self.save_checkpoint(fold)

        return best_auc
