import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchsummary import summary

#Standard Discriminator of CycleGAN
class Discriminator(nn.Module):
    def __init__(self,input,n_filter=64, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kernel_size = 4
        padding = 1
        self.first_conv = nn.Conv2d(input, n_filter, kernel_size=kernel_size, stride=2, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.second_conv = nn.Conv2d(n_filter,n_filter*2,kernel_size=kernel_size,stride=2,padding=padding)
        self.norm_1 =norm_layer(n_filter*2)

        self.third_conv = nn.Conv2d(n_filter*2,n_filter*4,kernel_size=kernel_size,stride=2,padding=padding)
        self.norm_2 = norm_layer(n_filter*4)

        self.fourth_conv = nn.Conv2d(n_filter*4,n_filter*8,kernel_size=kernel_size,stride=2,padding=padding)
        self.norm_3 = norm_layer(n_filter*8)

        self.last_conv = nn.Conv2d(n_filter*8, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self,input):
        y = self.first_conv(input)
        y = self.leaky_relu(y)

        y = self.second_conv(y)
        y = self.norm_1(y)
        y = self.leaky_relu(y)


        y = self.third_conv(y)
        y = self.norm_2(y)
        y = self.leaky_relu(y)


        y = self.fourth_conv(y)
        y = self.norm_3(y)
        y = self.leaky_relu(y)

        y = self.last_conv(y)

        return y
