import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.feature as KF

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = KF.DISK.from_pretrained('depth').eval()

    def run(self, img_tensor):
        """
        img_tensor: (1, 3, H, W)
        """
        # 1. Calculate new dimensions that are multiples of 16
        h, w = img_tensor.shape[2:]
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        
        # 2. Resize image
        img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.model(img_resized, n=1024, window_size=5)
            
            # 3. Rescale keypoints back to original image size
            kp = features[0].keypoints # (N, 2)
            kp[:, 0] *= (w / new_w)
            kp[:, 1] *= (h / new_h)
            
            des = features[0].descriptors # (N, 128)
            
        return kp, des