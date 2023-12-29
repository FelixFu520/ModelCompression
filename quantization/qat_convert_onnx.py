import onnx
from onnxsim import simplify
import torch

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

from utils import unet, wrapUnet


def convert(quantization_path, onnx_path):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()
    model = wrapUnet(nclass=2)

    # load the calibrated model
    state_dict = torch.load(quantization_path, map_location="cpu")
    model.load_state_dict(state_dict)
    # model.cuda()
    model.cpu()

    dummy_input = torch.randn(1, 1, 256, 256, device='cpu')
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=13, input_names=["actual_input_1"],
                      output_names=["output1"])
    onnx_model = onnx.load(onnx_path)
    optimized_model, check = simplify(onnx_model)
    onnx.save(optimized_model, onnx_path)


if __name__ == "__main__":
    quanization_path = "./checkpoints/wafer-qat/qat_epoch_39_loss_0.143339_ce_0.132922_dice_0.010417.pth"
    onnx_path = "./checkpoints/wafer-qat-calibrated.onnx"
    convert(quanization_path, onnx_path)
