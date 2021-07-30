import albumentations as A


def train_transform(level=0):
    assert 0 <= level <= 2, f'level must be in range [0, 2]'
    p = level * 0.25
    return A.Compose([  
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                     hue=0.2, p=0.3 + p),
        A.Cutout (num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.3 + p),
        # A.OneOf([
        #     A.OpticalDistortion(distort_limit=1.),
        #     A.GridDistortion(num_steps=5, distort_limit=1.),
        # ], p=0.3 + p),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1 ,
                          rotate_limit=10, border_mode=0, p=0.3 + p),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3 + p),
        A.Normalize(p=1.0),
    ])


def valid_transform():
    return A.Compose([
        A.Normalize(p=1.0),
    ])
