import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class UNet(nn.Module):
    
    def __init__(self,
                 encoder_name,
                 encoder_weights,
                 in_channels,
                 classes,
                 activation):
        super().__init__()
        self.arc = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
        
    def forward(self, images, masks=None):
        
        logits = self.arc(images)
        
        if masks != None:
            loss1 = DiceLoss(mode="binary")(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1+loss2
        
        return logits  
    
    