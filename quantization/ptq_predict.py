import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

from utils import unet

quant_modules.initialize()
quant_desc_input = QuantDescriptor(calib_method='histogram')  # ["max", "histogram"]
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

quant_modules.initialize()
model = unet(nclass=2)
model.load_state_dict(torch.load("./checkpoints/wafer-ptq-calibrated.pth", map_location="cpu"))


def run():
    # imgPath = "images/0000_Row006_Col036_00137_14.bmp"
    # dst = "images/0000_Row006_Col036_00137_14-pth-ptq.png"
    imgPath = "images/0500_Row023_Col050_00953_21.bmp"
    dst = "images/0500_Row023_Col050_00953_21-pth-ptq.png"
    img = cv2.imread(imgPath, 0)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = torch.from_numpy(img).to(device)  # torch.Size([1, 1, 256, 256])
    model.to(device)
    model.eval()
    out = model(img)  # torch.Size([1, 5, 256, 256])
    pred = F.softmax(out, dim=1)  # torch.Size([1, 5, 256, 256])
    pred = pred.detach().cpu().numpy()
    result = pred[0]  # c, h, w
    seg_result_ori = np.argmax(result, axis=0).astype(np.uint8)  # (512, 512)
    seg_result = (seg_result_ori != 0).astype(np.uint8)  # (h, w) # 目前先把所有前景类别当成一类显示
    seg_result_prob = seg_result * result.max(axis=0)
    cv2.imwrite(dst, seg_result_prob * 255)


if __name__ == "__main__":
    run()