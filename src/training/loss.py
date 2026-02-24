import torch
import torch.nn as nn
import json

def get_weighted_loss(weights_path, class_to_idx, device):
    """
    Creates a weighted CrossEntropyLoss using previously precomputed class weights.
    
    Args:
        weights_path: Path to JSON file with class weights
        class_to_idx: Dict mapping class names to integer indices
        device: torch device (cpu or cuda)
    
    Returns:
        nn.CrossEntropyLoss with class weights
    """
    # Load class weights from JSON
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)

    # Get num classes and initialize the weights as a vector of zeros
    num_classes = len(class_to_idx)
    weights = torch.zeros(num_classes)

    # Assign weights and move to device
    for class_name, idx in class_to_idx.items():
        weights[idx] = weights_dict[class_name]
    weights = weights.to(device)

    # Define weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=weights)

    return criterion