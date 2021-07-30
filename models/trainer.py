import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.log import log_info, log_warn
from utils.common import deprocess
from utils.plots import plot_data
from dataloader.augment import train_transform, valid_transform
from torch.utils.data import DataLoader
from dataloader.dataset import MyDataset
from tqdm import tqdm

LABELS = ['healthy', 'multiple_diseases', 'rust', 'scab']

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


class ClassificationTrainer:
    def __init__(self, model, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.weight_dir = os.path.join(self.args.work_dir, 'weights')
        self.augment_level = 0
        self.increase_augment_epochs = [int(c.strip()) for c in self.args.increase_augment_at.split(',')]
        os.makedirs(self.weight_dir, exist_ok=True)

    def train_one_epoch(self, train_loader):
        losses = AverageMeter()
        self.model.train()

        loop = tqdm(train_loader)
        preds = []
        y_true = []

        for image, labels in loop:
            self.optimizer.zero_grad()

            image = image.to(self.device)
            labels= labels.to(self.device)

            output = self.model(image)
            loss = self.criterion(output, labels)
            loss.backward()

            # Loss
            loss = loss.cpu().detach().numpy()
            losses.update(loss, image.shape[0])
            preds.append(F.softmax(output, dim=1).cpu().detach().numpy())
            y_true.append(labels.cpu().detach().numpy())

            self.optimizer.step()

        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)

        auc = roc_auc_score(y_true, preds, multi_class='ovr')
    
        return losses.avg, auc

    def val_one_epoch(self, loader):
        losses = AverageMeter()

        self.model.eval()

        preds = []
        y_true = []
        
        for image, labels in loader:
            image = image.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, labels)
            loss = loss.cpu().detach().numpy()
            losses.update(loss, image.shape[0])

            preds.append(F.softmax(output, dim=1).cpu().detach().numpy())
            y_true.append(labels.cpu().detach().numpy())
            
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)

        auc = roc_auc_score(y_true, preds, multi_class='ovr')

        return losses.avg, auc

    def get_loaders(self, df, fold):
        train_data = df[df.fold != fold].reset_index(drop=True)
        val_data = df[df.fold == fold].reset_index(drop=True)

        train_data = MyDataset(train_data, transform=train_transform(self.augment_level),
                               img_size=self.args.img_size)

        val_data= MyDataset(val_data, transform=valid_transform(),
                            img_size=self.args.img_size)
        
        train_loader = DataLoader(train_data,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True,
                                batch_size=self.args.batch_size)

        valid_loader = DataLoader(val_data,
                                shuffle=False,
                                num_workers=2,
                                batch_size=self.args.batch_size)
        
        return (train_loader, len(train_data)), (valid_loader, len(val_data))

    def save_checkpoint(self, fold, score):
        score = round(score, 2)
        path = f'{self.args.model}_fold{fold}_{score}.pth'
        old_ckp = glob.glob(os.path.join(
            self.weight_dir, path.replace(f'{score}.pth', '*')))

        for p in old_ckp:
            os.remove(p)

        torch.save(self.model.state_dict(), os.path.join(self.weight_dir, path))

    def load_checkpoint(self, fold):
        path = f'{self.args.model}_fold{fold}_*'
        files = glob.glob(os.path.join(self.weight_dir, path))
        if files:
            self.model.load_state_dict(torch.load(files[0], map_location='cpu'))
            log_info(f'Weight loaded from {files[0]}')
        else:
            log_info(f"No weight fold, pattern: {path}")

    def train(self, train_df, fold, start_epoch=0, best_score=0, in_progress=False):
        patient = 5 # Early stopping patient
        if self.args.resume and not in_progress:
            self.load_checkpoint(fold)

        if in_progress: # Training is in progress, but augment level increased
            log_info(f'Increase augmentation level from {self.augment_level} to {self.augment_level + 1}')
            self.augment_level + 1

        (train_loader, train_size), (valid_loader, val_size) = self.get_loaders(train_df, fold)

        if not in_progress: # Start training
            print(f'Train on {train_size} images, validate on {val_size} images')

            steps_per_epoch = train_size // self.args.batch_size
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, verbose=True, T_max=self.args.epochs * steps_per_epoch
            )

        # Plot examples
        for images, labels in train_loader:
            images = deprocess(images)
            labels = [LABELS[l] for l in labels]
            plot_data(images, labels,
                save_path=os.path.join(self.args.work_dir, 'train_sample.jpg'))
            break

        not_improve_e = 0
        for e in range(start_epoch, self.args.epochs):
            train_loss, train_score = self.train_one_epoch(train_loader)

            self.scheduler.step()

            val_loss, val_score = self.val_one_epoch(valid_loader)

            print((f'Epoch {e + 1}/{self.args.epochs}: Train loss {train_loss} - Train AUC {train_score}'
                   f' - Val loss {val_loss} - Val AUC {val_score}'))

            if val_score > best_score:
                log_info(f'Valid score improved from {best_score} to {val_score}')
                best_score = val_score
                self.save_checkpoint(fold, val_score)
            else:
                not_improve_e += 1

            if not_improve_e == patient:
                log_warn(f'Metric (Best = {best_score}) does not improve for {not_improve_e} epochs, stop training')

            if  self.increase_augment_epochs and e == self.increase_augment_epochs[0]:
                self.increase_augment_epochs.pop(0)
                return self.train(train_df, fold, start_epoch=e + 1, best_score=best_score, in_progress=True)

        return best_score
