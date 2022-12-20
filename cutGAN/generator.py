import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from slot_attention import SlotAttention
from torchsummary import summary

#Generator for CUT # TO DO: take number of resnet_blocks param and create blocks accordingly
class Generator(nn.Module):
    def __init__(self,input_dim,output_dim,number_of_resnet_blocks=6,n_filter=64,norm_layer=nn.InstanceNorm2d,use_dropout=False,use_bias=True,padding_type='reflect'):
        super(Generator,self).__init__()

        self.first_reflectionPad = nn.ReflectionPad2d(3)
        self.first_conv = nn.Conv2d(input_dim, n_filter, kernel_size=7, padding=0, bias=use_bias)
        self.first_norm = norm_layer(n_filter)
        self.first_relu = nn.ReLU(True)

        ########## DownSampling Conv Layers ##########
        self.second_conv = nn.Conv2d(n_filter, n_filter*2, kernel_size=3, stride=2,padding=1, bias=use_bias)
        self.second_norm = norm_layer(n_filter*2)
        self.second_relu = nn.ReLU(True)
    
        self.third_conv = nn.Conv2d(n_filter*2, n_filter*4, kernel_size=3, stride=2,padding=1, bias=use_bias)
        self.third_norm = norm_layer(n_filter*4)
        self.third_relu = nn.ReLU(True)
        ########## DownSampling Conv Layers ##########


        ########## Resnet Blocks  ##########
        self.resnet_block_1 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_2 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_3 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_4 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_5 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_6 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_7 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_8 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet_block_9 = ResnetBlock(n_filter*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)

        ########## Resnet Blocks  ##########


        ########## Upsampling Conv Layers ##########
        self.upsample_conv_1 = nn.ConvTranspose2d(n_filter * 4, int(n_filter * 4 / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        self.norm_upsample_1 = norm_layer(int(n_filter * 4 / 2))
        self.upsample_1_relu = nn.ReLU(True)

        self.upsample_conv_2 = nn.ConvTranspose2d(n_filter * 2, int(n_filter * 2 / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        self.norm_upsample_2 = norm_layer(int(n_filter * 2 / 2))
        self.upsample_2_relu = nn.ReLU(True)

        ########## Upsampling Conv Layers ##########

        
        ########## Final Conv Layers ##########
        self.last_reflectionPad = nn.ReflectionPad2d(3)
        self.last_conv = nn.Conv2d(n_filter, output_dim, kernel_size=7, padding=0)
        self.tanh = nn.Tanh()
        ########## Final Conv Layers ##########
        model = [self.first_reflectionPad, self.first_conv, self.first_norm, self.first_relu, self.second_conv, self.second_norm, self.second_relu, self.third_conv,
                        self.third_norm, self.third_relu, self.resnet_block_1, self.resnet_block_2, self.resnet_block_3, self.resnet_block_4, self.resnet_block_5,
                            self.resnet_block_6, self.resnet_block_7, self.resnet_block_8, self.resnet_block_9, self.upsample_conv_1, self.norm_upsample_1, self.upsample_1_relu,
                                self.upsample_conv_2, self.norm_upsample_2, self.upsample_2_relu, self.last_reflectionPad, self.last_conv, self.tanh]
        self.model = nn.Sequential(*model)


    def forward(self, input, layers=[], encode=False):
        if -1 in layers:
            layers.append(len(self.model))

        if len(layers) > 0:
            features = input
            feature_array = []
            for layer_id, layer in enumerate(self.model):
                features = layer(features)
                
                if layer_id in layers:
                    feature_array.append(features)
                else:
                    pass
                if layer_id == layers[-1] and encode:
                    return feature_array 

            return features, feature_array
        else:    
            y = self.model(input)

            return y

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out