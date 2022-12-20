import torch
import functools
import torch.nn as nn
import numpy as np
from torch.nn import init
from pyclbr import Class
from torch.optim import lr_scheduler
from slot_attention import SlotAttention
from torchsummary import summary

#Standard Discriminator of CUT
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

# MLP for the Discriminator defined as F
class MLP(nn.Module):
    def __init__(self,input_shape,output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.linear_1 = nn.Linear(input_shape,output_shape)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(output_shape,output_shape)


    def forward(self,x):

        y = self.linear_1(x)
        y = self.relu_1(y)
        y = self.linear_2(y)

        return y

# Discriminator of CUT defined as F (corresponding to patchSampleF in the original implementation)
class PatchSampleF(nn.Module):
    def __init__(self,num_channels=256):
        super().__init__()
        self.l2norm = Normalize(2)
        self.num_channels=num_channels
        self.mlp_init = True

    def forward(self,features, num_patches=64,patch_ids=None):
        return_ids = []
        return_feats = []
        
        #Creating MLPs for Patches 
        if self.mlp_init:
            for mlp_id, feature in enumerate(features):
                input_channel = feature.shape[1]
                mlp = MLP(input_channel,self.num_channels).cuda()
                setattr(self, 'mlp_%d' % mlp_id, mlp) # creating mlp objects for patches
            self.mlp_init = False
        
        # Patch pass
        for feature_id, feature in enumerate(features):
            B,H,W = feature.shape[0],feature.shape[2],feature.shape[3]
            reshaped_feature = feature.permute(0,2,3,1).flatten(1,2)

            if patch_ids is not None:
                patch_id = patch_ids[feature_id]
            else:
                patch_id = np.random.permutation(reshaped_feature.shape[1])
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]

            patch_id = torch.tensor(patch_id, dtype=torch.long, device=feature.device)
            x_sample = reshaped_feature[:, patch_id, :].flatten(0, 1)

            mlp = getattr(self, 'mlp_%d' % feature_id)
            x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            return_feats.append(x_sample)

        return return_feats, return_ids

## Taken from the original implementation
class Normalize(nn.Module):  
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out