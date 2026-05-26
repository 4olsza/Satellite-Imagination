import torch
import torch.nn as nn

# aim of the discriminator is to decide whether the generated picture is realistic or not and whether it matches the sketch

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1, # this setting reduces the size by half
                bias=False, # BatchNorm will turn it off anyway
                padding_mode="reflect" # copying edge pixels to avoid black stains
            ),

            # brings the results down to around zero to stabilise the training
            nn.BatchNorm2d(out_channels),

            # function of activation which allows positive values and lets negetive through by multiplaing them by a small number
            # in order to allow information to flow through the network
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)