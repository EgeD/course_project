import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchsummary import summary
import torch.nn.functional as F


#Standard Generator of CycleGAN
class Generator(nn.Module):
    def __init__(self,input_dim,output_dim,number_of_resnet_blocks=9,n_filter=64,norm_layer=nn.InstanceNorm2d,use_dropout=False,use_bias=True,padding_type='reflect'):
        super(Generator,self).__init__()

        # self.first_reflectionPad = nn.ReflectionPad2d(3)
        self.first_conv = nn.Conv2d(input_dim, n_filter, kernel_size=7, padding=0, bias=use_bias,)
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

        self.resnet_blocks = []
        for i in range(number_of_resnet_blocks):
            self.resnet_blocks.append(ResnetBlock(channel=n_filter*4, kernel=3, stride=1, padding=1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        ########## Resnet Blocks  ##########


        ########## Upsampling Content Conv Layers ##########
        self.content_upsample_conv_1 = nn.ConvTranspose2d(n_filter * 4, int(n_filter * 4 / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        self.content_norm_upsample_1 = norm_layer(int(n_filter * 4 / 2))
        self.content_upsample_1_relu = nn.ReLU(True)

        self.content_upsample_conv_2 = nn.ConvTranspose2d(n_filter * 2, int(n_filter * 2 / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        self.content_norm_upsample_2 = norm_layer(int(n_filter * 2 / 2))
        self.content_upsample_2_relu = nn.ReLU(True)
        self.content_upsample_conv_3 = nn.Conv2d(n_filter, 27, kernel_size=7, padding=0)

        ########## Upsampling Content Conv Layers ##########

        ########## Upsampling Attention Conv Layers ##########
        self.attention_upsample_conv_1 = nn.ConvTranspose2d(n_filter * 4, int(n_filter * 4 / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        self.attention_norm_upsample_1 = norm_layer(int(n_filter * 4 / 2))
        self.attention_upsample_1_relu = nn.ReLU(True)

        self.attention_upsample_conv_2 = nn.ConvTranspose2d(n_filter * 2, int(n_filter * 2 / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        self.attention_norm_upsample_2 = norm_layer(int(n_filter * 2 / 2))
        self.attention_upsample_2_relu = nn.ReLU(True)
        self.attention_upsample_conv_3 = nn.Conv2d(n_filter, 10, kernel_size=1, padding=0)

        ########## Upsampling Attention Conv Layers ##########
        # self.last_reflectionPad = nn.ReflectionPad2d(3)

        self.tanh = nn.Tanh()

    def forward(self, input):
        y = F.pad(input, (3, 3, 3, 3), 'reflect')

        y = self.first_conv(y)
        y = self.first_norm(y) 
        y = self.first_relu(y)

        ########## DownSampling Conv Layers ##########
        y = self.second_conv(y)
        y = self.second_norm(y)
        y = self.second_relu(y) 
    
        y = self.third_conv(y) 
        y = self.third_norm(y) 
        y = self.third_relu(y)
        ########## DownSampling Conv Layers ##########


        ########## Resnet Blocks  ##########

        y = self.resnet_blocks(y) 

        ########## Resnet Blocks  ##########


        ########## Upsampling Content Conv Layers ##########
        y_content = self.content_upsample_conv_1(y) 
        y_content = self.content_norm_upsample_1(y_content)
        y_content = self.content_upsample_1_relu(y_content)

        y_content = self.content_upsample_conv_2(y_content)
        y_content = self.content_norm_upsample_2(y_content)
        y_content = self.content_upsample_2_relu(y_content)

        y_content = F.pad(y_content, (3, 3, 3, 3), 'reflect')
        content = self.content_upsample_conv_3(y_content)
       
        image = self.tanh(content)
        
        image1 = image[:, 0:3, :, :]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]

        ########## Upsampling Content Conv Layers ##########

        ########## Upsampling Attention Conv Layers ##########

        y_attention = self.attention_upsample_conv_1(y) 
        y_attention = self.attention_norm_upsample_1(y_attention)
        y_attention = self.attention_upsample_1_relu(y_attention)

        y_attention = self.attention_upsample_conv_2(y_attention)
        y_attention = self.attention_norm_upsample_2(y_attention)
        y_attention = self.attention_upsample_2_relu(y_attention)

        attention = self.attention_upsample_conv_3(y_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        ########## Upsampling Attention Conv Layers ##########

        ########## Attention Cascading##########
        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, 3, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9

        output10 = input * attention10

        o= output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10

        ########## Attention Cascading##########
        return o, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention1,attention2,attention3, attention4, attention5, attention6, attention7, attention8,attention9,attention10, image1, image2,image3,image4,image5,image6,image7,image8,image9

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
class ResnetBlock(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(ResnetBlock, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()