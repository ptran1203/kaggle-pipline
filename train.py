import argparse
import pandas as pd
from models.trainer import Trainer
from models.classification import EfficientNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str)
    parser.add_argument('--n-classes', type=int)
    parser.add_argument('--model', type=str, default='tf_efficientnet_b0_ns')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)

    return parser.parse_args()

def main(args):
    df = pd.read_csv(args.data_csv)
    model = EfficientNet(args.model, args.n_classes)
    trainer = Trainer(model, args)
    trainer.train(df)


if __name__ == '__main__':
    main(parse_args())