import torch
import torch.nn as nn


def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out // 4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
    )
