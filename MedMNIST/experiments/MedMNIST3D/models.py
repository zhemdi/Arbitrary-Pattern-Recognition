'''
Adapted from kuangliu/pytorch-cifar .
'''

import torch.nn as nn
import torch.nn.functional as F


import sys
# sys.path.append('../../')


from ILPONet.source import InvLocalPatOrientConvolution as ILPO 




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout = 0.0, downsample_by_pooling = False):
        super(BasicBlock, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=conv_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        if dropout > 0.0:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = lambda x: x
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        if downsample_by_pooling and stride > 1:
            self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0)
        else:
            self.avgpool = lambda x: x

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=conv_stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes),
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes),
            
            )

    def forward(self, x):
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = self.bn2(self.conv2(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(self.dropout(out))
        return out

class ILPOBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, order = 3, so3_size = 3, stride=1, dropout = 0.0, downsample_by_pooling = False, pooling_type = 'softmax'):
        super(ILPOBasicBlock, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = ILPO(
            in_planes, planes, kernel_size=3, order = order, so3_size  = so3_size, stride=conv_stride, padding=1, bias=False, pooling_type = pooling_type )
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        if dropout > 0.0:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = lambda x: x
        self.conv2 = ILPO(planes, planes, kernel_size=3, order = order, so3_size  = so3_size, 
                               stride=1, padding=1, bias=False, pooling_type = pooling_type)
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        if downsample_by_pooling and stride > 1:
            self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0)
        else:
            self.avgpool = lambda x: x
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=conv_stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = self.bn2(self.conv2(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(self.dropout(out))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout = 0.0, downsample_by_pooling = False):
        super(Bottleneck, self).__init__()
        conv_stride =  1 if downsample_by_pooling else stride
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        if dropout > 0.0:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = lambda x: x
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=conv_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv3 = nn.Conv3d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)
        # self.bn3 = nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
        if downsample_by_pooling and stride > 1:
            self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0)
        else:
            self.avgpool = lambda x: x
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=conv_stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(self.dropout(out))
        return out
    

class ILPOBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, order = 3, so3_size = 3, stride=1, dropout = 0.0, downsample_by_pooling  = False, pooling_type = 'softmax'):
        super(ILPOBottleneck, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        if dropout > 0.0:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = lambda x: x
        self.conv2 = ILPO(planes, planes, kernel_size=3, order =order, so3_size  = so3_size, 
                               stride=conv_stride, padding=1, bias=False, pooling_type = pooling_type)
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv3 = nn.Conv3d(planes, self.expansion *
                               planes, kernel_size=1,  bias=False)
        
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)
        # self.bn3 = nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
        if downsample_by_pooling and stride > 1:
            self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0)
        else:
            self.avgpool = lambda x: x

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=conv_stride, bias=False),
                
                nn.BatchNorm3d(self.expansion*planes)
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(self.dropout(out))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, order = 3, so3_size  = 3, features = [64, 64, 128, 256, 512], strides = [1,1,2,2,2], in_channels=1, num_classes=2, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax'):
        super(ResNet, self).__init__()
        self.in_planes = features[0]
        if block == Bottleneck or block == BasicBlock:
            self.conv1 = nn.Conv3d(in_channels, features[0], kernel_size=3,
                                        stride=strides[0], padding=1, bias=False)
        elif block == ILPOBottleneck or block == ILPOBasicBlock:
            self.conv1 = ILPO(in_channels, features[0], kernel_size=3, order = order, so3_size  = so3_size,
                                        stride=strides[0], padding=1, bias=False)
        else:
            raise NotImplementedError('Block type not supported.')
        self.bn1 = nn.BatchNorm3d(features[0])
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.layer1 = self._make_layer(block, features[1], num_blocks[0], order = order, so3_size  = so3_size,  stride=strides[1], dropout = dropout, downsample_by_pooling = downsample_by_pooling, pooling_type =pooling_type)
        self.layer2 = self._make_layer(block, features[2], num_blocks[1], order = order, so3_size  = so3_size,  stride=strides[2], dropout = dropout, downsample_by_pooling = downsample_by_pooling, pooling_type =pooling_type)
        self.layer3 = self._make_layer(block, features[3], num_blocks[2], order = order, so3_size  = so3_size,  stride=strides[3], dropout = dropout, downsample_by_pooling = downsample_by_pooling, pooling_type =pooling_type)
        self.layer4 = self._make_layer(block, features[4], num_blocks[3], order = order, so3_size  = so3_size,  stride=strides[4], dropout = dropout, downsample_by_pooling = downsample_by_pooling, pooling_type =pooling_type)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(features[4] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, order, so3_size, stride, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i_s, stride in enumerate(strides):
            if block == Bottleneck or block == BasicBlock:
                layers.append(block(self.in_planes, planes, stride, downsample_by_pooling = downsample_by_pooling, dropout = dropout))
            elif block == ILPOBottleneck or block == ILPOBasicBlock:
                layers.append(block(self.in_planes, planes, order, so3_size, stride, dropout = dropout, downsample_by_pooling = downsample_by_pooling, pooling_type =pooling_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes, dropout = 0.0, downsample_by_pooling = False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, dropout = dropout, downsample_by_pooling=downsample_by_pooling)




def ILPOResNet18Small(in_channels, num_classes, order = 3, so3_size  = 3, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax'):
    return ResNet(ILPOBasicBlock, [2, 2, 2, 2], features=[4,4,4,4,4], strides = [1,1,1,1,1], in_channels=in_channels, num_classes=num_classes, order = order, so3_size  = so3_size, dropout = dropout, downsample_by_pooling=downsample_by_pooling, pooling_type = pooling_type)


def ILPOResNet18(in_channels, num_classes, order = 3, so3_size  = 3, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax'):
    return ResNet(ILPOBasicBlock, [2, 2, 2, 2], features=[8,8,8,8,8], strides = [1,1,1,1,1], in_channels=in_channels, num_classes=num_classes, order = order, so3_size  = so3_size, dropout = dropout, downsample_by_pooling=downsample_by_pooling, pooling_type = pooling_type)






def ResNet50(in_channels, num_classes, dropout = 0.0, downsample_by_pooling = False):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes, dropout = dropout, downsample_by_pooling=downsample_by_pooling)

def ILPOResNet50(in_channels, num_classes, order = 3, so3_size  = 3, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax'):
    return ResNet(ILPOBottleneck, [3, 4, 6, 3], features=[8,8,8,8,8], strides = [1,1,1,1,1], in_channels=in_channels, num_classes=num_classes, order = order, so3_size  = so3_size, dropout = dropout, downsample_by_pooling=downsample_by_pooling, pooling_type = pooling_type)

def ILPOResNet50Small(in_channels, num_classes, order = 3, so3_size  = 3, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax'):
    return ResNet(ILPOBottleneck, [3, 4, 6, 3], features=[4,4,4,4,4], strides = [1,1,1,1,1], in_channels=in_channels, num_classes=num_classes, order = order, so3_size  = so3_size, dropout = dropout, downsample_by_pooling=downsample_by_pooling, pooling_type = pooling_type)
