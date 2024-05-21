from monai.losses import DiceCELoss, DiceFocalLoss
from utility import *

from loss.soft_dice_ce_loss import soft_DC_and_CE_loss

def get_loss(config:dict):
    loss_name = config['Train']['loss']
    if loss_name == 'DiceCELoss':
        return DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, softmax=True)
    elif loss_name == 'DiceFocalLoss':
        return DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, softmax=True)
    elif loss_name == 'soft_DC_and_CE_loss':
        batch_dice = config['Train']['dataloader_geodesic']['batch_size']
        return soft_DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})



        