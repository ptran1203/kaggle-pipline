import pandas as pd
import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold


def main(args):
    train_df = pd.read_csv(args.input)
    LABELS = ['healthy', 'multiple_diseases', 'rust', 'scab']
    def onehot2label(row):
        idx = np.where(row[LABELS] == 1)[0][0]
        return idx

    # Create CSV contains: path, label
    train_df['path'] = train_df.image_id.apply(lambda x: os.path.join(args.image_dir, x + '.jpg'))
    train_df['label'] = train_df.apply(onehot2label, axis=1)
    # Group fold
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for i, (_, test_index) in enumerate(skf.split(train_df, train_df.label)):
        train_df.loc[test_index, 'fold'] = i

    train_df.to_csv(args.output)
    print(train_df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/content/dataset/train.csv')
    parser.add_argument('--output', type=str, default='data.csv')
    parser.add_argument('--image-dir', type=str, default='/content/dataset/images')

    args = parser.parse_args()

    main(args)