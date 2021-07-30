import argparse
from operator import index
import pandas as pd
import os
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.common import tta_predict
from sklearn.metrics import roc_auc_score
from dataloader.dataset import MyDataset
from dataloader.augment import valid_transform
from multiprocessing import cpu_count
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-dir', type=str, default='/content/exp')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--tta', action='store_true')

    return parser.parse_args()

def parse_model_info(weight_path):
    fname = weight_path.split(os.sep)[-1]
    model_name, *rest = fname.split('_fold')
    fold = int(rest[0].split('_')[0])

    return model_name, fold

def main(args):
    model_name, fold = parse_model_info(args.weight)

    print(f'Evaluate model {model_name} - fold {fold}')

    df = pd.read_csv(args.data_csv)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=False, num_classes=4).to(device)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    val_data = df[df.fold == fold].reset_index(drop=True)
    val_data= MyDataset(val_data, transform=valid_transform(),
                            img_size=args.img_size, return_path=True)
        
    valid_loader = DataLoader(val_data,
                            shuffle=False,
                            num_workers=cpu_count(),
                            batch_size=args.batch_size)

    bar = tqdm(valid_loader)
    preds = []
    y_true = []
    paths = []
    for path, image, labels in bar:
        image = image.to(device)

        with torch.no_grad():
            if args.tta:
                output = tta_predict(model, image)
            else:
                output = model(image)

        preds.append(F.softmax(output, dim=1).cpu().detach().numpy())
        y_true.append(labels.detach().cpu().numpy())
        paths.append(path)
    
    preds = np.concatenate(preds)
    y_true = np.concatenate(y_true)
    paths = np.concatenate(paths)

    auc = roc_auc_score(y_true, preds, multi_class='ovr')

    print(f'AUC {auc}')

    preds = [','.join([str(p) for p in pred]) for pred in preds]

    df = pd.DataFrame({
        'path': paths,
        'label': y_true,
        'pred': preds
    })
    df.to_csv(os.path.join(args.save_dir, 'prediction.csv'), index=False)
    print(f'Result saved to {args.save_dir}')



if __name__ == '__main__':
    main(parse_args())