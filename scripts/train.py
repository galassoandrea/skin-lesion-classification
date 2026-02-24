import pandas as pd
import numpy as np
import torch
from src.dataset.dataset import SkinLesionDataset
from src.models.classifiers import SkinLesionClassifier
from src.training.trainer import Trainer
from pathlib import Path
from src.dataset.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
from src.training.loss import get_weighted_loss
import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():    
    # Specify arguments
    ARGS = {
        'model_name': "facebook/deit-base-distilled-patch16-224",
        'num_classes': 7,
        'image_size': 224,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'num_workers': 4,
        'seed': 42,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }

    # Set seed for reproducibility
    set_seed(ARGS['seed'])

    # Define data paths
    csv_path = Path("data/splits")
    image_dir = Path("data/processed")

    # Create class_to_idx mapping from training data
    train_df = pd.read_csv(csv_path / "train.csv")
    unique_classes = sorted(train_df['dx'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

    # Load PyTorch Datasets
    train_dataset = SkinLesionDataset(
        csv_path=csv_path/"train.csv",
        image_dir=image_dir,
        transform=get_train_transforms(ARGS['image_size'])
    )
    val_dataset = SkinLesionDataset(
        csv_path=csv_path/"val.csv",
        image_dir=image_dir,
        transform=get_val_transforms(ARGS['image_size'])
    )

    # Define PyTorch Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=ARGS['batch_size'],
        shuffle=True,
        num_workers=ARGS['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ARGS['batch_size'],
        shuffle=False,
        num_workers=ARGS['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Load pretrained model with custom classification head
    print("\nInitializing model...")
    model = SkinLesionClassifier(model_name=ARGS['model_name'])
    model = model.to(ARGS['device'])

    # Count and print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize loss function with class weights
    criterion = get_weighted_loss(
        weights_path=csv_path / "class_weights.json",
        class_to_idx=class_to_idx,
        device=ARGS['device']
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ARGS['learning_rate']
    )

    # Initialize Trainer object
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=ARGS['device']
    )

    # Train the model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train(num_epochs=ARGS['num_epochs'])
    
    print("\nTraining completed succesfully.")

if __name__ == "__main__":
    main()





