import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

def compute_metrics(outputs, labels, num_classes=7):
    """
    Compute accuracy, F1 and AUC for multi-class classification.
    
    Args:
        outputs: Model logits, shape (batch_size, num_classes)
        labels: Ground truth labels, shape (batch_size,)
        num_classes: Number of classes
    
    Returns:
        Dict with 'accuracy', 'f1', 'auc'
    """
    # Convert to numpy
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Get predictions (argmax of logits)
    predictions = np.argmax(outputs, axis=1)
    
    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Compute macro F1 (average F1 across classes)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Compute macro AUC (one-vs-rest)
    try:
        # Convert logits to probabilities
        probabilities = softmax(outputs, axis=1)
        
        # One-hot encode labels for AUC calculation
        labels_onehot = label_binarize(labels, classes=range(num_classes))
        
        auc = roc_auc_score(labels_onehot, probabilities, average='macro', multi_class='ovr')
    except ValueError:
        # Can happen if a class is missing in the batch
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc
    }