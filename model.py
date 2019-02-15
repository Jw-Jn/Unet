import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes, bn=False):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downStep1 = downStep(1, 64, firstLayer=True, bn=bn)
        self.downStep2 = downStep(64, 128, bn=bn)
        self.downStep3 = downStep(128, 256, bn=bn)
        self.downStep4 = downStep(256, 512, bn=bn)
        self.downStep5 = downStep(512, 1024, bn=bn)
        
        self.upStep1 = upStep(1024, 512, bn=bn)
        self.upStep2 = upStep(512, 256, bn=bn)
        self.upStep3 = upStep(256, 128, bn=bn)
        self.upStep4 = upStep(128, 64, withReLU=False, bn=bn)

        self.conv = nn.Conv2d(64, n_classes, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # todo
        x1 = self.downStep1(x)
        x2 = self.downStep2(x1)
        x3 = self.downStep3(x2)
        x4 = self.downStep4(x3)
        # x4 = self.dropout(x4)
        x5 = self.downStep5(x4) 
        

        x = self.upStep1(x5, x4)
        # print(x.shape)
        x = self.upStep2(x, x3)
        # print(x.shape)
        x = self.upStep3(x, x2)
        # print(x.shape)
        x = self.upStep4(x, x1)
        # print(x.shape)

        x = self.conv(x)
        # print(x.shape)

        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, firstLayer=False, bn=False):
        super(downStep, self).__init__()
        # todo
        self.bn = bn
        self.firstLayer = firstLayer
        self.conv_bn = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=0),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=0),
            nn.BatchNorm2d(outC),
            nn.ReLU())

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=0),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # todo
        if not self.firstLayer:
            x = self.maxpool(x)

        if self.bn:
            x = self.conv_bn(x)
        else:
            x = self.conv(x)

        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True, bn=False):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.bn = bn
        self.withReLU = withReLU
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=0),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=0),
            nn.BatchNorm2d(outC),
            nn.ReLU())

        self.conv_bn = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=0),
            nn.BatchNorm2d(outC),
            nn.Conv2d(outC, outC, 3, padding=0),
            nn.BatchNorm2d(outC))

        self.conv_relu = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=0),
            nn.ReLU())

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=0),
            nn.Conv2d(outC, outC, 3, padding=0))

        self.upsampling = nn.ConvTranspose2d(inC, outC, 2, 2)

    def forward(self, x, x_down):
        # todo
        x = self.upsampling(x)
        
        # input is CHW
        py = x_down.size()[2] - x.size()[2]
        px = x_down.size()[3] - x.size()[3]

        x_down = x_down[:, :, py//2:x_down.shape[2]-py//2, px//2:x_down.shape[3]-px//2]

        x = torch.cat([x_down, x], dim=1) #512, 1024

        if self.withReLU and self.bn: # not last layer, with bn
            x = self.conv_bn_relu(x)

        elif self.withReLU:  # not last layer, without bn
            x = self.conv_relu(x)
        
        elif self.bn: #last layer, with bn
            x = self.conv_bn(x)

        else:  # last layer, without bn
            x = self.conv(x)
        
        return x