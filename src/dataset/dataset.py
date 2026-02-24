from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import numpy as np

class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion classification.
    
    Args:
        csv_path: Path to train/val/test CSV with columns [image_id, dx, lesion_id]
        image_dir: Directory containing processed images
        transform: transform pipeline
    """

    def __init__(self, csv_path, image_dir, transform=None):

        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Get unique diagnostic categories and enumerate them to get an index for each class
        categories = sorted(self.df['dx'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(categories)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_id = row['image_id']
        dx = row['dx']

        # Load image and apply transforms
        image_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        transformed = self.transform(image=image)
        image = transformed['image']

        # Convert class to index label
        label = self.class_to_idx[dx]

        return image, label