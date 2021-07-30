import matplotlib.pyplot as plt
import math

def plot_data(imgs, labels, figsize=(10, 10),save_path=''):
    '''
    Plot imgs and labels pair

    @Inputs:
        - imgs: array shape hxwxc
        - labels: array of labels
    '''
    cols = 4
    if len(imgs) < 5:
        cols = 2

    rows = math.ceil(len(imgs) / cols)

    fig = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.text(0.7, 10, labels[i], color='k', backgroundcolor='white')
        plt.imshow(imgs[i])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
