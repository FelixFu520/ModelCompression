import datetime
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import logging
import random

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

from utils import segDataset, unet, bootstrapped_cross_entropy2d, DiceLoss, init_logging

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

# -------------------------- Hyperparameter1 ------------------------------
experment_name = 'wafer-qat'
PRETRAIN_MODEL = None
Start_epoch = 0
SEED = 0
NUM_EPOCH = 40
VERBOSE_STEP = 40
BATCH_SIZE = 64  # 64 #256 #256#256
lr0 = 1e-2
wd0 = 1e-5
K = 10000
optimType = "Adam"

# -------------------------- setting ------------------------------
if not os.path.exists("checkpoints/{}".format(experment_name)):
    os.makedirs("checkpoints/{}".format(experment_name))
log_file = './checkpoints/{}/log.log'.format(experment_name)
file_mode = 'w'
init_logging(log_file=log_file, file_mode=file_mode, overwrite_flag=True, log_level=logging.DEBUG)
# -------------------------- Dataset ------------------------------
train_path = "../Datasets/wafer/data/train"
train_set = segDataset(train_path, transforms=None)
test_path = "../Datasets/wafer/data/val"
val_set = segDataset(test_path, transforms=None)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True,
                              pin_memory=False)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=True,
                            pin_memory=False)

# -------------------------- Reproducibility ------------------------------
# torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

quant_modules.initialize()
model = unet(nclass=2)
model.load_state_dict(torch.load("checkpoints/wafer-ptq-calibrated.pth", map_location="cpu"))
model.cuda()
diceLoss = DiceLoss().to(device)

# -------------------------- Optimizer loading --------------------------
if optimType == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=wd0)
elif optimType == "SGD":
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr0, momentum=0.9, dampening=0.5, weight_decay=wd0,
                                nesterov=False)
else:
    print("cannot support optimizer:{}".format(optimType))
    exit(-1)

lr_schduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3)


# lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)


# ------------------------- train ---------------------------------------
def train(model, epoch):
    model.train()
    running_loss = []
    running_ce_loss = []
    running_dice_loss = []
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, data in loop:
        img, label = data
        img = img.to(device, non_blocking=True)
        # label = label.astype(np.int64)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(img)
        dice_loss = diceLoss(output, label)
        ce_loss = bootstrapped_cross_entropy2d(input=output, target=label, K=K)
        all_loss = dice_loss + ce_loss
        all_loss.backward()

        optimizer.step()
        running_loss.append(all_loss.item())
        running_ce_loss.append(ce_loss.item())
        running_dice_loss.append(dice_loss.item())
        loop.set_description("Loss: {:.6f}".format(np.mean(running_loss)))

        if i % VERBOSE_STEP == 0 and i != 0:
            logging.info("E:{:02}/{:02} P:{:03}/{:03} All:{:.6f} ce:{:.6f} dice:{:.6f}".format(epoch,
                                                                                               NUM_EPOCH,
                                                                                               i,
                                                                                               len(train_dataloader),
                                                                                               np.mean(running_loss),
                                                                                               np.mean(running_ce_loss),
                                                                                               np.mean(
                                                                                                   running_dice_loss)))
            running_loss = []
            running_ce_loss = []
            running_dice_loss = []
    return model


# ------------------------- eval ---------------------------------------
def eval(model, epoch):
    model.eval()
    all_loss_list = []
    ce_loss_list = []
    dice_loss_list = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
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

    logging.info("{} Test Results {}".format("-" * 20, "-" * 20))
    _all_loss = np.mean(all_loss_list)
    _ce_loss = np.mean(ce_loss_list)
    _dice_loss = np.mean(dice_loss_list)

    logging.info(
        "eval E:{:02}/{:03} all:{:.6f} ce:{:.6f} dice:{:.6f}".format(epoch, NUM_EPOCH, _all_loss, _ce_loss, _dice_loss))
    logging.info("-" * 56)
    return _all_loss, _ce_loss, _dice_loss


if __name__ == "__main__":
    for epoch in range(Start_epoch, NUM_EPOCH):
        logging.info("****************lr:{}****************".format(optimizer.param_groups[0]['lr']))
        model = train(model, epoch)
        all_loss, ce_loss, dice_loss = eval(model, epoch)
        lr_schduler.step(all_loss)
        torch.save(model.state_dict(),
                   "./checkpoints/{}/qat_epoch_{:02}_loss_{:.6f}_ce_{:.6f}_dice_{:.6f}.pth".format(experment_name,
                                                                                                   epoch, all_loss,
                                                                                                   ce_loss, dice_loss),
                   _use_new_zipfile_serialization=False)
    print("Training finished")
