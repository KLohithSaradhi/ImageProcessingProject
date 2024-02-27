import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
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

class ReconstructionLoss(_Loss):

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, y):
        x = torch.flatten(x)
        y = torch.flatten(y)

        loss = torch.mean(torch.abs(x - y))

        return loss
    
class VGGPerceptualLoss(nn.Module):
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

class SSIM(_Loss):
    def __init__(self, K1 = 0.01, K2 = 0.01, K3 = 0.01, L = 2, alpha = 1, beta = 1, gamma = 1):
        super(SSIM, self).__init__()

        self.C1 = (K1 * L) ** 2
        self.C2 = (K2 * L) ** 2
        self.C3 = (K3 * L) ** 2

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y):

        mu_x = torch.mean(x)
        mu_y = torch.mean(y)

        l_xy = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)

        sigma_x = torch.sqrt(torch.var(x))
        sigma_y = torch.sqrt(torch.var(y))

        c_xy = (2 * sigma_x * sigma_y + self.C2) / (sigma_x ** 2 + sigma_y ** 2 + self.C2)

        cov = torch.mean((x - mu_x) * (y - mu_y))

        s_xy = (cov + self.C3) / (sigma_x * sigma_y + self.C3)

        ssim = l_xy ** self.alpha * c_xy ** self.beta * s_xy ** self.gamma

        return ssim

class LocalSSIM(_Loss):
    def __init__(self, slice = -1):
        super(LocalSSIM, self).__init__()

        self.SSIMLoss = SSIM()
        self.slice = slice

    def forward(self, x, y):
        x_chunks = x.shape[2] // self.slice
        y_chunks = x.shape[3] // self.slice    

        SSIM = 0

        for i in range(x_chunks - 1):
            for j in range(y_chunks - 1):
                x_chunk = x[:, :, i * self.slice : (i + 1) * self.slice, j * self.slice : (j + 1) * self.slice]
                y_chunk = y[:, :, i * self.slice : (i + 1) * self.slice, j * self.slice : (j + 1) * self.slice]

                SSIM += self.SSIMLoss(x_chunk, y_chunk)

        SSIM = SSIM / (x_chunks * y_chunks)

        return SSIM   
        
class TotalLoss(_Loss):
    def __init__(self, lambda1 = 0.1, lambda2 = 0.1):
        super(TotalLoss, self).__init__()

        self.recon = ReconstructionLoss()
        self.perceptual = VGGPerceptualLoss()
        self.ssim = LocalSSIM(slice = 4)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def forward(self, groundTruths, outputs):

        LOSS = 0

        for scale in range(len(groundTruths)):
            loss = self.recon(outputs[scale], groundTruths[scale])
            loss += self.lambda1 * self.perceptual(outputs[scale], groundTruths[scale])
            loss += self.lambda2 * self.ssim(outputs[scale], groundTruths[scale])

            LOSS += loss * (1 / (2 ** scale))
        
        return LOSS