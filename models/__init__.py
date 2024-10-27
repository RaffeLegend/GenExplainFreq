from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .reconstruct_models import Reconstruction
import torch.nn as nn

from .resnet_model import ResModel
from .classification_head import Classification


VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
]


# def get_model(name):
#     assert name in VALID_NAMES
#     if name.startswith("Imagenet:"):
#         return ImagenetModel(name[9:]) 
#     elif name.startswith("CLIP:"):
#         return CLIPModel(name[5:])  
#     else:
#         assert False 
def get_model(name):
    return DualBranchNetwork(name)

# Combined model with both branches
class DualBranchNetwork(nn.Module):
    def __init__(self, name):
        super(DualBranchNetwork, self).__init__()
        
        # Initialize the classification branch (CLIP)
        # self.classification_branch = CLIPModel(name[5:])
        self.backbone = nn.Sequential(*list(ResModel.children())[:-2])
        # Initialize the reconstruction branch (ResNet50)
        self.reconstruction_branch = nn.Sequential(*list(Reconstruction.children())[:-2])
        self.classification_branch = Classification()
    
    def forward(self, x):
        # Branch 1: Classification (CLIP)
        feature = self.backbone(x)
        classification_logits = self.classification_branch(feature)
        
        # Branch 2: Image reconstruction (ResNet50)
        reconstructed_image = self.reconstruction_branch(feature)
        
        return classification_logits, reconstructed_image