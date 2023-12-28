import onnx
from onnxsim import simplify

import torch

from utils import unet, wrapUnet


def convert(pth_path, onnx_path):
    model = wrapUnet(nclass=2)

    # load the calibrated model
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict)
    # model.cuda()

    dummy_input = torch.randn(1, 1, 256, 256, device='cpu')
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=13, input_names=["actual_input_1"],
                      output_names=["output1"])
    onnx_model = onnx.load(onnx_path)
    optimized_model, check = simplify(onnx_model)
    onnx.save(optimized_model, onnx_path)


if __name__ == "__main__":
    pth_path = "./checkpoints/wafer-train/epoch_380_loss_0.182091_ce_0.161683_dice_0.020408.pth"
    onnx_path = "./checkpoints/wafer-train.onnx"
    convert(pth_path, onnx_path)
