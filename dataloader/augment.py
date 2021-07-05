import albumentations as A


def train_transform():
    return A.Compose([  
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.),
            A.GridDistortion(num_steps=5, distort_limit=1.),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=14, sat_shift_limit=14 ,
                             val_shift_limit=0, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1 ,
                          rotate_limit=10, border_mode=0, p=0.5),
        A.Normalize(p=1.0),
    ])


def valid_transform():
    return A.Compose([
        A.Normalize(p=1.0),
    ])
