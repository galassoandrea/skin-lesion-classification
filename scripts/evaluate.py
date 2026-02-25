"""
Evaluation script for skin lesion classifier.
Run from project root: python -m scripts.evaluate
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset.dataset import SkinLesionDataset
from src.dataset.transforms import get_val_transforms
from src.models.classifiers import SkinLesionClassifier
from src.training.metrics import compute_metrics

# Specify arguments
ARGS = {
    'model_name': "facebook/deit-base-distilled-patch16-224",
    'num_classes': 7,
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    'checkpoint_path': 'checkpoints/best_model.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    
    # Define data paths
    data_dir = Path("data")
    csv_path = data_dir / "splits"
    image_dir = data_dir / "processed"
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Create class_to_idx mapping
    train_df = pd.read_csv(csv_path / "train.csv")
    unique_classes = sorted(train_df['dx'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Load PyTorch Dataset
    print("\nLoading test dataset...")
    test_dataset = SkinLesionDataset(
        csv_path=csv_path / "test.csv",
        image_dir=image_dir,
        transform=get_val_transforms(ARGS['image_size']),
        # class_to_idx=class_to_idx
    )

    # Define PyTorch Dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=ARGS['batch_size'],
        shuffle=False,
        num_workers=ARGS['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model without pretrained weights
    print("\nLoading model...")
    model = SkinLesionClassifier(
        model_name=ARGS['model_name'],
        num_classes=ARGS['num_classes'],
        pretrained=False
    )

    # Load custom weights into the model
    checkpoint = torch.load(ARGS['checkpoint_path'], map_location=ARGS['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(ARGS['device'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation F1: {checkpoint['best_val_f1']:.4f}")
    
    # Run inference
    print("\nRunning inference on test set...")
    all_outputs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(ARGS['device'])
            outputs = model(images)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels)
            all_predictions.append(torch.argmax(outputs, dim=1).cpu())
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_outputs, all_labels, ARGS['num_classes'])
    
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1']:.4f}")
    print(f"Macro AUC: {metrics['auc']:.4f}")
    
    # Classification report
    print("\nPer-class metrics:")
    report = classification_report(
        all_labels.numpy(),
        all_predictions.numpy(),
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save classification report
    report_dict = classification_report(
        all_labels.numpy(),
        all_predictions.numpy(),
        target_names=class_names,
        output_dict=True
    )
    
    with open(outputs_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nClassification report saved to {outputs_dir / 'classification_report.json'}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels.numpy(), all_predictions.numpy())
    plot_confusion_matrix(cm, class_names, outputs_dir / "confusion_matrix.png")
    
    # Save summary
    summary = {
        'test_accuracy': float(metrics['accuracy']),
        'test_f1': float(metrics['f1']),
        'test_auc': float(metrics['auc']),
        'num_test_samples': len(test_dataset),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'best_val_f1': float(checkpoint['best_val_f1']),
    }
    
    with open(outputs_dir / "test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTest summary saved to {outputs_dir / 'test_results.json'}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()