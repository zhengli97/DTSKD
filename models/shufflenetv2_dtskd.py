"""shufflenetv2 in pytorch
[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x


class ShuffleNetV2(nn.Module):

    def __init__(self, ratio=1, num_classes=100):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], num_classes)

        laterals, upsample = [], []
        for i in range(3):
            laterals.append(self._lateral(out_channels[i], 1024))
        for i in range(2, 4):
            upsample.append(self._upsample(channels=1024))
        self.laterals = nn.ModuleList(laterals)
        self.upsample = nn.ModuleList(upsample)
        self.fuse_1 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.fuse_2 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.fuse_3 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.up_conv = nn.Sequential(
            nn.Conv2d(1024, 1024,kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.fc_b1 = nn.Linear(1024, num_classes)
        self.fc_b2 = nn.Linear(1024, num_classes)
        self.fc_b3 = nn.Linear(1024, num_classes)

    def _upsample(self, channels=512):
        layers = []
        # layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(channels, channels,
                                      kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _lateral(self, input_size, output_size=512):
        layers = []
        layers.append(nn.Conv2d(input_size, output_size,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        # layers.append(nn.BatchNorm2d(input_size))
        layers.append(nn.Conv2d(output_size, output_size,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        s_out1 = self.stage2(x)  # 116,16,16
        s_out2 = self.stage3(s_out1)  # 232,8,8
        s_out3 = self.stage4(s_out2)  # 464,4,4
        s_out4 = self.conv5(s_out3)  # 1024,4,4

        out = F.adaptive_avg_pool2d(s_out4, (1,1))
        logits_out = out 
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # t_out4 = self.laterals[3](s_out4)  # 128,1024,4,4
        t_out4 = logits_out

        upsample3 = self.up_conv(t_out4)
        t_out3 = torch.cat([(upsample3 + self.laterals[2](s_out3)), upsample3], dim=1)
        t_out3 = self.fuse_3(t_out3)

        upsample2 = self.upsample[1](t_out3)
        t_out2 = torch.cat([(upsample2 + self.laterals[1](s_out2)), upsample2], dim=1)
        t_out2 = self.fuse_2(t_out2)

        upsample1 = self.upsample[0](t_out2)
        t_out1 = torch.cat([(upsample1 + self.laterals[0](s_out1)), upsample1], dim=1)
        t_out1 = self.fuse_1(t_out1)

        t_out3 = torch.flatten(t_out3, 1)
        t_out3 = self.fc_b3(t_out3)

        t_out2 = torch.flatten(t_out2, 1)
        t_out2 = self.fc_b2(t_out2)

        t_out1 = torch.flatten(t_out1, 1)
        t_out1 = self.fc_b1(t_out1)

        return out, t_out1, t_out2, t_out3

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)


def shufflenetv2(**kwargs):
    return ShuffleNetV2(**kwargs)


if __name__ == '__main__':
    input = torch.ones([128, 3, 32, 32])
    model = shufflenetv2()
    output = model(input)
    print("test")
