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
            nn.InstanceNorm2d(out_channels),

            # function of activation which allows positive values and lets negetive through by multiplaing them by a small number
            # in order to allow information to flow through the network
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # initial layer - skipping BatchNorm at first to keep the input colors (adding mathematics will distort them)
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, # channels from two pictures - sketch and satellite,
                out_channels=features[0], # 64 channels
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        # core layer
        self.core = nn.Sequential(
            # channels: 64 -> 128, size of the picture reduced by half
            CNNBlock(features[0], features[1], stride=2),

            # channels: 128 -> 256, size reduced by a half
            CNNBlock(features[1], features[2], stride=2),

            # channels: 256 -> 512, size of the picture is small enough - no need to reduce it
            CNNBlock(features[2], features[3], stride=1)
        )

        # final layer - reduces 512 channels to just 1 (true/false rating matrix)
        self.final = nn.Conv2d(
            in_channels=features[3],
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode="reflect"
        )
    
    def forward(self, x, y):
        # x: input (map's sketch)
        # y: judged picture (satellite view)

        # creating one block from two pictures
        merged = torch.cat([x,y], dim=1) # dim=1 -> bonding along the channels

        # putting merged block into discriminator
        initial_out = self.initial(merged)
        core_out = self.core(initial_out)
        final_out = self.final(core_out)

        return final_out