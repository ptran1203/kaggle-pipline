import numpy as np
import torch
import torchvision.transforms.functional as tvf
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

def onehot_encode(labels, n_classes=None):
    if isinstance(labels, np.ndarray):
        onehot = np.zeros((labels.size, n_classes))
        onehot[np.arange(labels.size), labels] = 1
    else:
        # Torch tensor
        size = labels.size()[0]
        onehot = torch.zeros((size, n_classes))
        onehot[torch.arange(size), labels] = 1

    return onehot


def deprocess(imgs):
    '''
    Channel first -> channel last then de-normalize
    '''
    imgs = imgs.permute(0, 2, 3, 1)
    imgs = imgs * STD + MEAN
    return imgs


def tta(images):
    '''
    Augment images with:
        - left-right flip
        - up-down flip
        - Central crop
    '''
    img_size = images.shape[-2]
    crop_size = img_size * 0.75

    return [
        images,
        torch.flip(images, dims=[1]), # ud
        torch.flip(images, dims=[2]), # lr
        F.interpolate(tvf.center_crop(images, crop_size), size=img_size),
    ]


def tta_predict(model, images):
    '''
    Test time augmentation
    '''
    outputs = []
    for imgs in tta(images):
        output = model(imgs)
        outputs.append(output)

    mean = torch.mean(torch.stack(outputs), 0)
    return mean