import argparse
import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from utils.common import tta_predict
from dataloader.dataset import MyDataset
from dataloader.augment import valid_transform
from multiprocessing import cpu_count
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str)
    parser.add_argument('--weights', type=str, help='Separate by whitespace')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=str, default='/content/submission.csv')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--tta', action='store_true')

    return parser.parse_args()

def get_model_name(weight_path):
    fname = weight_path.split(os.sep)[-1]
    model_name, *rest = fname.split('_fold')

    return model_name


def load_models(weights, device):
    n_classes = 4 # Doesn't change during the competition
    models = []
    for weight in weights.split(' '):
        model_name = get_model_name(weight)
        model = timm.create_model(model_name, pretrained=False, num_classes=4)
        model.load_state_dict(torch.load(weight, map_location='cpu'))
        models.append(model.to(device))

    return models


def main(args):
    if args.tta:
        args.output = args.output.replace('.csv', '') + '_tta.csv'
    df = pd.read_csv(args.data_csv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = load_models(args.weights, device)

    print(f'Ensemble of {len(models)} models')

    data = MyDataset(df, transform=valid_transform(), img_size=args.img_size)
    
    loader = DataLoader(data,
                        shuffle=False,
                        num_workers=cpu_count(),
                        batch_size=args.batch_size)

    bar = tqdm(loader)

    sub_df = pd.read_csv('/content/dataset/sample_submission.csv')
    preds = []

    for image, _ in bar:
        image = image.to(device)
        outputs = []
        for model in models:
            with torch.no_grad():
                if args.tta:
                    output = tta_predict(model, image)
                else:
                    output = model(image)
            outputs.append(output)

        if len(outputs) > 1:
            outputs = torch.mean(torch.stack(outputs), 0)
        else:
            outputs = outputs[0]

        preds.append(F.softmax(outputs, dim=1).cpu().detach().numpy())

    preds = np.concatenate(preds)
    sub_df.loc[:, 1:] = preds
    sub_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main(parse_args())