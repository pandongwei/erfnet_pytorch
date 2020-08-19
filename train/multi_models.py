# setup a multitask learning model with backbone ERFNet for Pytorch
# August 2020
# Dongwei Pan
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class DownsamplerBlock_simplified (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput, (3, 3), stride=1, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.pool(output)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downblock_1 = DownsamplerBlock_simplified(3,16)

        self.downblock_2 = nn.ModuleList()
        self.downblock_2.append(DownsamplerBlock_simplified(16,64))
        for x in range(0, 3):    # 5 times-> 2 times
           self.downblock_2.append(non_bottleneck_1d(64, 0.03, 1))

        self.downblock_3 = nn.ModuleList()
        self.downblock_3.append(DownsamplerBlock_simplified(64,128))

        for x in range(0, 1):    # 2 times -> 1 times
            self.downblock_3.append(non_bottleneck_1d(128, 0.3, 2))
            self.downblock_3.append(non_bottleneck_1d(128, 0.3, 4))
            self.downblock_3.append(non_bottleneck_1d(128, 0.3, 8))
            self.downblock_3.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input, predict=False):
        output = self.downblock_1(input)
        shortcut_1 = output
        for layer in self.downblock_2:
            output = layer(output)
        shortcut_2 = output
        for layer in self.downblock_3:
            output = layer(output)

        return output, shortcut_1, shortcut_2


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder_traversability (nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.upblock_1 = nn.ModuleList()

        self.upblock_1.append(UpsamplerBlock(128,64))
        self.upblock_1.append(non_bottleneck_1d(64, 0, 1))
        self.upblock_1.append(non_bottleneck_1d(64, 0, 1))

        self.upblock_2 = nn.ModuleList()
        self.upblock_2.append(UpsamplerBlock(128,16))
        self.upblock_2.append(non_bottleneck_1d(16, 0, 1))
        self.upblock_2.append(non_bottleneck_1d(16, 0, 1))

        self.upblock_3 = nn.ModuleList()
        self.upblock_3.append(nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True))


    def forward(self, input, shortcut_1, shortcut_2):
        output = input

        for layer in self.upblock_1:
            output = layer(output)
        output = torch.cat([output,shortcut_2], dim=1)

        for layer in self.upblock_2:
            output = layer(output)
        output = torch.cat([output,shortcut_1], dim=1)

        for layer in self.upblock_3:
            output = layer(output)

        return output


class Decoder_depth(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.upblock_1 = nn.ModuleList()

        self.upblock_1.append(UpsamplerBlock(128, 64))
        self.upblock_1.append(non_bottleneck_1d(64, 0, 1))
        self.upblock_1.append(non_bottleneck_1d(64, 0, 1))

        self.upblock_2 = nn.ModuleList()
        self.upblock_2.append(UpsamplerBlock(128, 16))
        self.upblock_2.append(non_bottleneck_1d(16, 0, 1))
        self.upblock_2.append(non_bottleneck_1d(16, 0, 1))

        self.upblock_3 = nn.ModuleList()
        self.upblock_3.append(nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True))

    def forward(self, input, shortcut_1, shortcut_2):
        output = input

        for layer in self.upblock_1:
            output = layer(output)
        output = torch.cat([output, shortcut_2], dim=1)

        for layer in self.upblock_2:
            output = layer(output)
        output = torch.cat([output, shortcut_1], dim=1)

        for layer in self.upblock_3:
            output = layer(output)

        return output

class Classifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1))
        self.classifier.append(nn.BatchNorm2d(64, eps=1e-03))
        self.classifier.append(nn.Dropout2d(0.3))
        self.classifier.append(nn.ReLU())
        self.classifier = nn.Sequential(*self.classifier)

        self.linear = nn.Linear(64, num_classes)

    def forward(self, input):
        output = self.classifier(input)
        output = output.mean(3).mean(2)
        output = self.linear(output)
        return output

class Multi_models(nn.Module):
    def __init__(self, num_classes=14):  #use encoder to pass pretrained encoder
        super().__init__()

        self.encoder = Encoder()

        self.decoder_traversability = Decoder_traversability(1)
        self.decoder_depth = Decoder_depth(1)
        self.classifier = Classifier(num_classes)

    def forward(self, input):
        feature_map, shortcut_1, shortcut_2 = self.encoder(input)
        output_traversability = self.decoder_traversability(feature_map, shortcut_1, shortcut_2)
        output_depth = self.decoder_depth(feature_map, shortcut_1, shortcut_2)
        output_class = self.classifier(feature_map)

        return output_traversability, output_depth, output_class