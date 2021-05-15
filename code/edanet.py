"""
# This python file is an implementation of proposed efficient dual attention networks (EDA)
@author: Md Mostafa Kamal Sarker
@ email: m.kamal.sarker@gmail.com
@ Date: 23.05.2017
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import PAM_Module, CAM_Module, DuaAttn
from itertools import chain
from efficientnet import EfficientNet

class EDANet(nn.Module):
    """ efficient dual attention networks (EDA)"""
    def __init__(self, num_classes=3, inplanes=1):
        super(EDANet, self).__init__()
        
        ## DuaAttn blocks 
        self.dua2= DuaAttn(40)
        self.dua3= DuaAttn(112)
        self.dua4= DuaAttn(192)
        self.dua5= DuaAttn(320)

        ## stem to the efficientnet (initial block)
        self._conv_stem = nn.Conv2d(inplanes, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(32)
        
        ## EfficinetNet pretrained backbone
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        ## delete the final FC layer of EfficinetNet
        del self.model._fc
        ## In case of using different backbones of EfficinetNet
        # self._conv_final = nn.Conv2d(1280, 320, kernel_size=3, bias=False)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Linear(1280, num_classes)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, inputs):
        bs = inputs.size(0)   
        ## define backbone   
        backbone = self.model
        # Convolution layers
        x = backbone._swish(self._bn0(self._conv_stem(inputs)))
        # print("stem:", x.size())
        x = backbone.extract_features_block1(x)
        # x =  self.dua1(x)
        # print("bloc1:", x.size())
        x = backbone.extract_features_block2(x)
        x =  self.dua2(x)
        # print("bloc2:", x.size())
        x = backbone.extract_features_block3(x)
        x =  self.dua3(x)
        # print("bloc3:", x.size())
        x = backbone.extract_features_block4(x)
        x =  self.dua4(x)
        # print("bloc4:", x.size())
        x = backbone.extract_features_block5(x)
        x =  self.dua5(x)
        # print("bloc5:", x.size())
        ## Head
        x = backbone._swish(backbone._bn1(backbone._conv_head(x)))
        # print("head:", x.size())       
        # x = self._conv_final(x)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = backbone._dropout(x)
        x = self._fc(x)
        x = self.softmax (x)
        return x

