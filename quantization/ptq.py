import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

from utils import collect_stats, compute_amax, unet, segDataset, bootstrapped_cross_entropy2d, DiceLoss

# Model
quant_modules.initialize()
quant_desc_input = QuantDescriptor(calib_method='histogram')  # ["max", "histogram"]
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = unet(nclass=2)
print("Loading FP32 ckpt")
model.load_state_dict(torch.load("./checkpoints/wafer-train/epoch_380_loss_0.182091_ce_0.161683_dice_0.020408.pth"))
model.to(device)
diceLoss = DiceLoss().to(device)
K = 10000
# Data
print("Preparing data")

# -------------------------- Dataset ------------------------------
BATCH_SIZE = 128
train_path = "../Datasets/wafer/data/train"
train_set = segDataset(train_path, transforms=None)
test_path = "../Datasets/wafer/data/val"
val_set = segDataset(test_path, transforms=None)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True,
                              pin_memory=False)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=True,
                            pin_memory=False)


def eval(model):
    model.eval()
    all_loss_list = []
    ce_loss_list = []
    dice_loss_list = []

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            img, label = data
            img = img.to(device, non_blocking=True)
            # label = label.astype(np.int64)
            label = label.to(device, non_blocking=True)
            output = model(img)
            dice_loss = diceLoss(output, label)
            ce_loss = bootstrapped_cross_entropy2d(input=output, target=label, K=K)
            all_loss = dice_loss + ce_loss
            all_loss_list.append(all_loss.item())
            ce_loss_list.append(ce_loss.item())
            dice_loss_list.append(dice_loss.item())
    _all_loss = np.mean(all_loss_list)
    _ce_loss = np.mean(ce_loss_list)
    _dice_loss = np.mean(dice_loss_list)
    return _all_loss, _ce_loss, _dice_loss


# It is a bit slow since we collect histograms on CPU
print("Collecting histograms")
with torch.no_grad():
    collect_stats(model, train_dataloader, num_batches=2)
    compute_amax(model, method="percentile", percentile=99.99, strict=False)

# Evaluate
criterion = nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
_all_loss, _ce_loss, _dice_loss = eval(model)
print("PTQ metric all:{:.6f} ce:{:.6f} dice:{:.6f}".format(_all_loss, _ce_loss, _dice_loss))

# Save the model
torch.save(model.state_dict(), "./checkpoints/wafer-ptq-calibrated.pth")
print("Done")