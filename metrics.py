import torch
import numpy as np

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = torch.nn.BCELoss()
        self.bce_weight = bce_weight
        
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return bce * self.bce_weight + dice * (1 - self.bce_weight)

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate true positives, false negatives, and false positives
        tp = (pred_flat * target_flat).sum()
        fn = (target_flat * (1-pred_flat)).sum()
        fp = ((1-target_flat) * pred_flat).sum()
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)
        return 1 - tversky

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
        
    def forward(self, pred, target):
        tversky_loss = self.tversky(pred, target)
        return torch.pow(tversky_loss, self.gamma)

def dice_coefficient(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target, smooth=1e-5):
    """Calculate IoU/Jaccard index"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

def precision(pred, target, smooth=1e-5):
    """Calculate precision"""
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.view(-1)
    
    true_positives = (pred_flat * target_flat).sum()
    predicted_positives = pred_flat.sum()
    
    return (true_positives + smooth) / (predicted_positives + smooth)

def recall(pred, target, smooth=1e-5):
    """Calculate recall/sensitivity"""
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.view(-1)
    
    true_positives = (pred_flat * target_flat).sum()
    actual_positives = target_flat.sum()
    
    return (true_positives + smooth) / (actual_positives + smooth)

def specificity(pred, target, smooth=1e-5):
    """Calculate specificity"""
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.view(-1)
    
    true_negatives = ((1 - pred_flat) * (1 - target_flat)).sum()
    actual_negatives = (1 - target_flat).sum()
    
    return (true_negatives + smooth) / (actual_negatives + smooth)

def f1_score(pred, target, smooth=1e-5):
    """Calculate F1 score"""
    prec = precision(pred, target, smooth)
    rec = recall(pred, target, smooth)
    
    return 2 * (prec * rec) / (prec + rec + smooth)

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate all metrics"""
    # Apply threshold to predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate metrics
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    prec = precision(pred, target)
    rec = recall(pred, target)
    spec = specificity(pred, target)
    f1 = f1_score(pred, target)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': prec.item(),
        'recall': rec.item(),
        'specificity': spec.item(),
        'f1': f1.item()
    }

class DeepSupervisionLoss(torch.nn.Module):
    def __init__(self, loss_fn, weights=None):
        """
        A wrapper for deep supervision loss
        
        Args:
            loss_fn: Base loss function to use
            weights: Weights for each output level (default: [1.0, 0.8, 0.6, 0.4, 0.2])
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights if weights is not None else [1.0, 0.8, 0.6, 0.4, 0.2]
        
    def forward(self, preds, target):
        if not isinstance(preds, list):
            # If no deep supervision, just use the single output
            return self.loss_fn(preds, target)
        
        # Calculate loss for each level of deep supervision
        loss = 0
        for i, pred in enumerate(preds):
            if i < len(self.weights):
                weight = self.weights[i]
                # For deep supervision outputs, repeat target if needed
                if i > 0:
                    loss += weight * self.loss_fn(pred, target)
                else:
                    loss += weight * self.loss_fn(pred, target)
        
        return loss

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_dice = BCEDiceLoss(smooth=1e-5, bce_weight=0.5)
        self.focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5)
        
    def forward(self, pred, target):
        return 0.5 * self.bce_dice(pred, target) + 0.5 * self.focal_tversky(pred, target)
    
# In metrics.py, create a specialized loss function:
class WeightedFocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, small_lesion_boost=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.small_lesion_boost = small_lesion_boost
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate lesion size
        lesion_size = target_flat.sum() / target_flat.size(0)
        
        # Boost weight for small lesions
        size_weight = torch.clamp(self.small_lesion_boost / (lesion_size + 1e-5), min=1.0, max=4.0)
        
        # Calculate true positives, false negatives, and false positives
        tp = (pred_flat * target_flat).sum()
        fn = (target_flat * (1-pred_flat)).sum()
        fp = ((1-target_flat) * pred_flat).sum()
        
        # Apply weighted Tversky
        tversky = (tp + 1e-5) / (tp + self.alpha*fn*size_weight + self.beta*fp + 1e-5)
        
        # Apply focal component to focus on harder examples
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        
        return focal_tversky