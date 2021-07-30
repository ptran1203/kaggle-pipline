import argparse
import pandas as pd
import os
import timm
from models.trainer import ClassificationTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str)
    parser.add_argument('--n-classes', type=int, default=4)
    parser.add_argument('--model', type=str, default='tf_efficientnet_b0_ns')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--work-dir', type=str, default='/content/exp')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--increase-augment-at', type=str, default='8,16',
        help='Increase level of augmentation at epochs, seperated by comma')

    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.data_csv)
    os.makedirs(args.work_dir, exist_ok=True)
    model = timm.create_model(args.model, pretrained=True, num_classes=args.n_classes)
    trainer = ClassificationTrainer(model, args)
    trainer.train(df, args.fold)


if __name__ == '__main__':
    main(parse_args())