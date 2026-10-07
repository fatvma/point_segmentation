import torch
import torch.nn as nn
import torchvision

class DecoderBlock(nn.Module):
    """One 'growing' step: upsamples, then fuses in the matching skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet34(weights=weights)

        # the "shrinking" half -- borrowed straight from ResNet
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # the "growing" half -- I built this myself
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)

        self.head = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        p = self.pool(s0)
        s1 = self.layer1(p)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        d0 = self.dec0(d1, None)

        return self.head(d0)
    
if __name__ == "__main__":
    model = ResNetUNet(n_classes=5, pretrained=False)  # False = skip downloading weights for this quick test
    x = torch.randn(1, 3, 256, 256)  # 1 image, 3 color channels, 256x256 pixels
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape, "-- expect (1, 5, 256, 256)")