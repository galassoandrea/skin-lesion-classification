"""
Apply augmentation transforms to the images to make
training more robust and less prone to overfit.
Notice that we apply augmentation only during training,
while during inference (validation/test), we only normalize the images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=224):
    """
    Transform pipeline for training.
    Applies random transformations to increase data diversity.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=20, 
            val_shift_limit=10, 
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(image_size=224):
    """
    Simple pipeline for validation/test.
    No augmentation, just resize and normalize.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])