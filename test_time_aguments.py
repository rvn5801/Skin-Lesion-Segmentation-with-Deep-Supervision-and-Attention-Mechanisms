import torch
import albumentations as A
import numpy as np

def tta_predict(model, img, device, n_augmentations=5):
    """Predict with test-time augmentation"""
    predictions = []
    
    # Original prediction
    with torch.no_grad():
        pred = model(img.to(device))
        predictions.append(pred.cpu())
    
    # Augmented predictions
    tta_transforms = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=10, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=1.0)
    ]
    
    for transform in tta_transforms:
        # Convert tensor to numpy for albumentations
        img_np = img.squeeze(0).permute(1,2,0).cpu().numpy()
        augmented = transform(image=img_np)
        aug_img = torch.from_numpy(augmented['image']).permute(2,0,1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(aug_img)
            
            # Reverse augmentation for prediction
            pred_np = pred.squeeze(0).cpu().numpy()
            
            # Apply inverse transforms
            if isinstance(transform, A.HorizontalFlip):
                pred_np = np.flip(pred_np, axis=1)
            elif isinstance(transform, A.VerticalFlip):
                pred_np = np.flip(pred_np, axis=0)
            # For other transforms, we'd need more complex inverse operations
            
            pred_tensor = torch.from_numpy(pred_np).unsqueeze(0)
            predictions.append(pred_tensor)
    
    # Average predictions
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred