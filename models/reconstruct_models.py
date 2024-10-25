from PIL import Image
import torch.nn as nn
from resnet import resnet50

    
# Reconstruction branch using ResNet50
class ReconstructionModel(nn.Module):
    def __init__(self):
        super(ReconstructionModel, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = resnet50(pretrained=True)
        
        # Remove the fully connected layer (keep feature extractor)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output is (batch_size, 2048, 7, 7)
        
        # Upsampling module for image reconstruction
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # Upsample 1
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),   # Upsample 2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # Upsample 3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # Upsample 4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),      # Final output (RGB)
            nn.Sigmoid()  # Normalize output to [0,1]
        )
        
    def forward(self, x):
        # Extract features using ResNet50 backbone
        features = self.backbone(x)  # Output is (batch_size, 2048, 7, 7)
        
        # Reconstruct image
        reconstructed_image = self.upsample(features)
        return reconstructed_image