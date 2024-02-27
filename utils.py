import torch
import torch.nn as nn
import torchvision




class SelectiveResidualBlock(nn.Module):
    def __init__(self, channels = 64):
        super(SelectiveResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))


    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.norm2(x)

        out = self.alpha * residual + self.beta * x

        out = self.ReLU(out)

        return out
    
class EncodingBlock(nn.Module):
    def __init__(self, channels = 64):
        super(EncodingBlock, self).__init__()

        self.block1 = SelectiveResidualBlock(channels)
        self.block2 = SelectiveResidualBlock(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.conv(x)
        out = self.ReLU(x)

        return out

class DecodingBlock(nn.Module):
    def __init__(self, channels = 64):
        super(DecodingBlock, self).__init__()

        self.pixelShuffle = nn.PixelShuffle(2)
        self.block1 = SelectiveResidualBlock(channels//4)
        self.block2 = SelectiveResidualBlock(channels//4)

    def forward(self, x, side_concat):
        x = self.pixelShuffle(x)
        x = self.block1(x)
        out = self.block2(x)

        out = torch.cat((out, side_concat), dim = 1)

        return out

class ReconstructionLoss(nn.modules.loss):

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, y):
        x = torch.flatten(x)
        y = torch.flatten(y)

        loss = torch.mean(torch.abs(x - y))

        return loss
    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class TotalLoss(nn.modules.loss):
    def __init__(self):
        super(TotalLoss, self).__init__()

        

        


