from PIL import Image
import torch.nn as nn
from models.resnet import resnet50
from .FreqFusion import FreqFusion
from SegNeXt.mmseg.models.decode_heads.ham_head import LightHamHead

    
# Reconstruction branch using ResNet50
class ReconstructionModel(nn.Module):
    def __init__(self, ham_channels):
        super(ReconstructionModel, self).__init__()
        
        # Load pre-trained ResNet50
        self.head = LightHamHead()
        
    def forward(self, x):
        # Extract features using ResNet50 backbone
        features = self.backbone(x)  # Output is (batch_size, 2048, 7, 7)
        
        # Reconstruct image
        reconstructed_image = self.upsample(features)
        return reconstructed_image
    
class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()

        self.head = LightHamHead()
        
    def forward(self, x):
        # Extract features using ResNet50 backbone
        features = self.head(x)  # Output is (batch_size, 2048, 7, 7)
        
        # Reconstruct image
        reconstructed_image = self.upsample(features)
        return reconstructed_image