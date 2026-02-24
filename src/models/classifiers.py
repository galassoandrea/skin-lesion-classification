import torch.nn as nn
from transformers import DeiTForImageClassification, DeiTConfig

class SkinLesionClassifier(nn.Module):
    """
    Vision Transformer for skin lesion classification.
    Uses a pretrained DeiT backbone with a custom classification head.
    
    Args:
        model_name: Name of pretrained model from huggingface transformers
        num_classes: Number of diagnostic categories (7 for HAM10000)
        pretrained: Whether to load ImageNet pretrained weights
    """

    def __init__(self, model_name="facebook/deit-base-distilled-patch16-224", num_classes=7, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if pretrained:
            # Load pretrained model and modify classification head
            self.model = DeiTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Allows replacing the head
            )
        else:
            # Load from scratch
            config = DeiTConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = DeiTForImageClassification(config)

    def forward(self, x):
        """        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        outputs = self.model(x)
        return outputs.logits

    def freeze_backbone(self):
        """
        Freeze all layers except the classification head.
        Useful for initial training phase.
        """
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreeze all parameters for full fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True