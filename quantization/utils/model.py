import numpy as np
import torch.nn as nn
import torch.utils.data
import torch

__all__ = ['unet', 'wrapUnet']


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class down_conv(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class conv_bn(nn.Module):
    """
    conv_bn Block
    """
    def __init__(self, in_ch, out_ch, stride=1, padding=1):
        super(conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),)
    def forward(self, x):
        x = self.conv(x)
        return x
    
class conv_bn_relu(nn.Module):
    """
    conv_bn_relu Block
    """
    def __init__(self, in_ch, out_ch, stride=1, padding=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x

class res_down(nn.Module):
    """
    res_down Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(res_down, self).__init__()
        self.conv1 = conv_block(in_ch=in_ch, out_ch=out_ch)
        self.conv2 = conv_bn(in_ch=out_ch, out_ch=out_ch,stride=1, padding=0)
        
        self.conv3 = conv_bn(in_ch=in_ch, out_ch=out_ch,stride=2, padding=0)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(x)
        e4 = torch.add(e2,e3)
        e4 = self.act1(e4)
        return e4


class res_horizon(nn.Module):
    """
    res_horizon Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(res_horizon, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        e1 = self.conv(x)
        e2 = torch.add(x,e1)
        e2 = self.act(e2)
        return e2
        
class horizon(nn.Module):
    """
    horizon Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(horizon, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        e1 = self.conv(x)
        e2 = torch.add(x,e1)
        e2 = self.act(e2)
        return e2
        


def batchSoftmax(x,axis=1):
    # 计算指数
    exp_x = np.exp(x)
    # 对第二维度进行softmax操作
    exp_sum = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_output = exp_x / exp_sum
    return softmax_output

class unet(nn.Module):
    def __init__(self, nclass=5):
        super(unet, self).__init__()
        self.conv1 = down_conv(in_ch=1, out_ch=32)
        
        self.conv2 = conv_block(in_ch=32, out_ch=24)
        self.conv3 = conv_bn(in_ch=24, out_ch=24,stride=1, padding=0)
        self.conv4 = conv_bn(in_ch=32, out_ch=24,stride=2, padding=0)
        self.act1 = nn.ReLU(inplace=True)
        
        self.res_down1 = res_down(in_ch=32, out_ch=24)
        self.res_down2 = res_down(in_ch=24, out_ch=56)
        self.res_down3 = res_down(in_ch=56, out_ch=152)
        
        self.res_horizon1 = res_horizon(in_ch=152, out_ch=152)
        self.res_horizon2 = res_horizon(in_ch=152, out_ch=152)
        self.res_horizon3 = res_horizon(in_ch=152, out_ch=152)
        
        self.res_down4 = res_down(in_ch=152, out_ch=368)
        
        
        self.res_horizon4 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon5 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon6 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon7 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon8 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon9 = res_horizon(in_ch=368, out_ch=368)
        
        self.conv5 = conv_bn_relu(in_ch=368, out_ch=128,stride=1, padding=0)
        self.horizon1 = horizon(in_ch=128, out_ch=128)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = conv_bn_relu(in_ch=280, out_ch=64, stride=1, padding=0)
        self.horizon2 = horizon(in_ch=64, out_ch=64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = conv_bn_relu(in_ch=120, out_ch=32, stride=1, padding=0)
        self.horizon3 = horizon(in_ch=32, out_ch=32)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv8 = conv_bn_relu(in_ch=56, out_ch=16, stride=1, padding=0)
        self.horizon4 = horizon(in_ch=16, out_ch=16)
        
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv9 = conv_bn_relu(in_ch=16, out_ch=16, stride=1, padding=0)
        self.conv10 = conv_bn(in_ch=16, out_ch=nclass, stride=1, padding=0)
        
    def forward(self,x):
        e = (x- 127.5)/127.5     #torch.Size([32,1,256,256])
        e1 = self.conv1(e)       #torch.Size([32, 32, 128, 128])
        
        e2 = self.res_down1(e1) #torch.Size([32, 24, 64, 64])
        e3 = self.res_down2(e2) #torch.Size([32, 56, 32, 32])
        e4 = self.res_down3(e3) #torch.Size([32, 152, 16, 16])

        e5 = self.res_horizon1(e4)#torch.Size([32, 152, 16, 16])
        e6 = self.res_horizon2(e5)#torch.Size([32, 152, 16, 16])
        e7 = self.res_horizon3(e6)#torch.Size([32, 152, 16, 16])
        
        e8 = self.res_down4(e7) #torch.Size([32, 368, 8, 8])
        
        e9 = self.res_horizon4(e8)#torch.Size([32, 368, 8, 8])
        e10 = self.res_horizon5(e9)#torch.Size([32, 368, 8, 8])
        e11 = self.res_horizon6(e10)#torch.Size([32, 368, 8, 8])
        e12 = self.res_horizon7(e11)#torch.Size([32, 368, 8, 8])
        e13 = self.res_horizon8(e12)#torch.Size([32, 368, 8, 8])
        e14 = self.res_horizon9(e13)#torch.Size([32, 368, 8, 8])
        e15 = self.conv5(e14) #torch.Size([32, 128, 8, 8])
        e16 = self.horizon1(e15)#torch.Size([32, 128, 8, 8])
        e17 = self.up1(e16)    #torch.Size([32, 128, 16, 16])
        e18 = torch.cat((e7, e17), dim=1) #torch.Size([32, 280, 16, 16])
        e19 = self.conv6(e18) #torch.Size([32, 64, 16, 16])
        e20 = self.horizon2(e19)#torch.Size([32, 64, 16, 16])
        e21 = self.up2(e20)    #torch.Size([32, 64, 32, 32])
        e22 = torch.cat((e3, e21), dim=1)
        e23 = self.conv7(e22) #torch.Size([32, 32, 32, 32])
        e24 = self.horizon3(e23)#torch.Size([32, 32, 32, 32])
        e25 = self.up3(e24)    #torch.Size([32, 32, 64, 64])
        e26 = torch.cat((e2, e25), dim=1)#torch.Size([32, 56, 64, 64])
        e27 = self.conv8(e26) #torch.Size([32, 16, 64, 64])
        e28 = self.horizon4(e27)#torch.Size([32, 16, 64, 64])
        e29 = self.up4(e28)  #torch.Size([32, 16, 256, 256])
        e30 = self.conv9(e29)
        e31 = self.conv10(e30)#torch.Size([32, 4, 256, 256])
        #e32 = e31.permute(0,2,3,1) #torch.Size([32, 256, 256, 4])
        #pred = nn.functional.softmax(pred, dim=1)
        return e31
        
class wrapUnet(nn.Module):
    def __init__(self, nclass=5):
        super(wrapUnet, self).__init__()
        self.conv1 = down_conv(in_ch=1, out_ch=32)

        self.conv2 = conv_block(in_ch=32, out_ch=24)
        self.conv3 = conv_bn(in_ch=24, out_ch=24, stride=1, padding=0)
        self.conv4 = conv_bn(in_ch=32, out_ch=24, stride=2, padding=0)
        self.act1 = nn.ReLU(inplace=True)

        self.res_down1 = res_down(in_ch=32, out_ch=24)
        self.res_down2 = res_down(in_ch=24, out_ch=56)
        self.res_down3 = res_down(in_ch=56, out_ch=152)

        self.res_horizon1 = res_horizon(in_ch=152, out_ch=152)
        self.res_horizon2 = res_horizon(in_ch=152, out_ch=152)
        self.res_horizon3 = res_horizon(in_ch=152, out_ch=152)

        self.res_down4 = res_down(in_ch=152, out_ch=368)

        self.res_horizon4 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon5 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon6 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon7 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon8 = res_horizon(in_ch=368, out_ch=368)
        self.res_horizon9 = res_horizon(in_ch=368, out_ch=368)

        self.conv5 = conv_bn_relu(in_ch=368, out_ch=128, stride=1, padding=0)
        self.horizon1 = horizon(in_ch=128, out_ch=128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = conv_bn_relu(in_ch=280, out_ch=64, stride=1, padding=0)
        self.horizon2 = horizon(in_ch=64, out_ch=64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = conv_bn_relu(in_ch=120, out_ch=32, stride=1, padding=0)
        self.horizon3 = horizon(in_ch=32, out_ch=32)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv8 = conv_bn_relu(in_ch=56, out_ch=16, stride=1, padding=0)
        self.horizon4 = horizon(in_ch=16, out_ch=16)

        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv9 = conv_bn_relu(in_ch=16, out_ch=16, stride=1, padding=0)
        self.conv10 = conv_bn(in_ch=16, out_ch=nclass, stride=1, padding=0)

    def forward(self, x):
        e = (x - 127.5) / 127.5  # torch.Size([32,1,256,256])
        e1 = self.conv1(e)  # torch.Size([32, 32, 128, 128])

        e2 = self.res_down1(e1)  # torch.Size([32, 24, 64, 64])
        e3 = self.res_down2(e2)  # torch.Size([32, 56, 32, 32])
        e4 = self.res_down3(e3)  # torch.Size([32, 152, 16, 16])

        e5 = self.res_horizon1(e4)  # torch.Size([32, 152, 16, 16])
        e6 = self.res_horizon2(e5)  # torch.Size([32, 152, 16, 16])
        e7 = self.res_horizon3(e6)  # torch.Size([32, 152, 16, 16])

        e8 = self.res_down4(e7)  # torch.Size([32, 368, 8, 8])

        e9 = self.res_horizon4(e8)  # torch.Size([32, 368, 8, 8])
        e10 = self.res_horizon5(e9)  # torch.Size([32, 368, 8, 8])
        e11 = self.res_horizon6(e10)  # torch.Size([32, 368, 8, 8])
        e12 = self.res_horizon7(e11)  # torch.Size([32, 368, 8, 8])
        e13 = self.res_horizon8(e12)  # torch.Size([32, 368, 8, 8])
        e14 = self.res_horizon9(e13)  # torch.Size([32, 368, 8, 8])
        e15 = self.conv5(e14)  # torch.Size([32, 128, 8, 8])
        e16 = self.horizon1(e15)  # torch.Size([32, 128, 8, 8])
        e17 = self.up1(e16)  # torch.Size([32, 128, 16, 16])
        e18 = torch.cat((e7, e17), dim=1)  # torch.Size([32, 280, 16, 16])
        e19 = self.conv6(e18)  # torch.Size([32, 64, 16, 16])
        e20 = self.horizon2(e19)  # torch.Size([32, 64, 16, 16])
        e21 = self.up2(e20)  # torch.Size([32, 64, 32, 32])
        e22 = torch.cat((e3, e21), dim=1)
        e23 = self.conv7(e22)  # torch.Size([32, 32, 32, 32])
        e24 = self.horizon3(e23)  # torch.Size([32, 32, 32, 32])
        e25 = self.up3(e24)  # torch.Size([32, 32, 64, 64])
        e26 = torch.cat((e2, e25), dim=1)  # torch.Size([32, 56, 64, 64])
        e27 = self.conv8(e26)  # torch.Size([32, 16, 64, 64])
        e28 = self.horizon4(e27)  # torch.Size([32, 16, 64, 64])
        e29 = self.up4(e28)  # torch.Size([32, 16, 256, 256])
        e30 = self.conv9(e29)
        e31 = self.conv10(e30)  # torch.Size([32, 4, 256, 256])
        # e32 = e31.permute(0,2,3,1) #torch.Size([32, 256, 256, 4])
        # pred = nn.functional.softmax(pred, dim=1)
        output = nn.functional.softmax(e31, dim=1)
        return output


if __name__ == "__main__":
    model = unet(nclass=5)
    input = torch.randn(32,1,256,256)
    out = model(input)
    print("out:",out.shape) #torch.Size([32, 5, 256, 256])
    
