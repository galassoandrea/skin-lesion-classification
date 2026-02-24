import torch
from tqdm import tqdm
from pathlib import Path
from src.training.metrics import compute_metrics

class Trainer:
    """
    Handles training and validation loops for the skin lesion classifier.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        criterion: Loss function
        optimizer: Optimizer (e.g., Adam)
        device: torch device
        num_classes: Number of classes
        checkpoint_dir: Directory to save model checkpoints
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_classes=7,
        checkpoint_dir='checkpoints',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            all_outputs.append(outputs.detach())
            all_labels.append(labels.detach())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_outputs, all_labels, self.num_classes)
        
        return epoch_loss, metrics
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)
        
        # Compute validation metrics
        val_loss = running_loss / len(self.val_loader.dataset)
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_outputs, all_labels, self.num_classes)
        
        return val_loss, metrics
    
    def train(self, num_epochs):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            # Save best model based on validation F1
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"New best model saved! (F1: {self.best_val_f1:.4f})")
            
            # Save latest checkpoint
            self.save_checkpoint('latest_model.pth', epoch, val_metrics)
        
        print(f"\nTraining complete! Best validation F1: {self.best_val_f1:.4f}")
    
    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'metrics': metrics,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        return checkpoint
