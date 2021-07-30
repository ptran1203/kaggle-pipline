import numpy as np


def acc(labels, preds):
    '''
    Calculate accuracy

    @Inputs:
        labels: categirical labels
        preds: logit or categorical
    '''

    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=1)

    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    return (labels == preds).mean()
