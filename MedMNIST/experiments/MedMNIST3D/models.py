'''
Adapted from kuangliu/pytorch-cifar .
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


from ILPONet import InvLocalPatOrientConvolution as ILPO 
from EquiLoPO import (
    EquiLoPOConvolution as ELPO,
    SE3BatchNorm,
    SE3Dropout,
    AvgPoolSE3,
    SO3Softmax,
    SO3GlobalActivation,
    SO3LocalActivation,
    Retyper,
)


class BasicBlock(nn.Module):
    """
    A basic block for ResNet, consisting of two 3D convolutional layers.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0, downsample_by_pooling=False):
        """
        Initialize the BasicBlock.

        Parameters:
        - in_planes: Number of input planes.
        - planes: Number of output planes.
        - stride: Stride for the convolutional layer.
        - dropout: Dropout rate.
        - downsample_by_pooling: Whether to downsample by pooling.
        """
        super(BasicBlock, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=conv_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(dropout, inplace=True) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0) if downsample_by_pooling and stride > 1 else nn.Identity()
        self.shortcut = self._make_shortcut(in_planes, planes, conv_stride)

    def _make_shortcut(self, in_planes, planes, stride):
        """
        Create a shortcut layer to match dimensions when necessary.

        Parameters:
        - in_planes: Number of input planes.
        - planes: Number of output planes.
        - stride: Stride for the convolutional layer.

        Returns:
        - A sequential model representing the shortcut.
        """
        if stride != 1 or in_planes != self.expansion * planes:
            return nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )
        return nn.Sequential()

    def forward(self, x):
        """
        Forward pass for the BasicBlock.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor after passing through the block.
        """
        identity = self.shortcut(x)
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.avgpool(identity)
        out = F.relu(out)
        return out
    



class ILPOBasicBlock(nn.Module):
    """
    A basic block using InvLocalPatOrientConvolution (ILPO) for ResNet-like architectures.

    Attributes:
        conv1 (ILPO): First ILPO convolutional layer.
        bn1 (nn.BatchNorm3d): Batch normalization after the first ILPO layer.
        dropout (nn.Dropout3d or Identity): Dropout or identity function for regularizing.
        conv2 (ILPO): Second ILPO convolutional layer.
        bn2 (nn.BatchNorm3d): Batch normalization after the second ILPO layer.
        avgpool (nn.AvgPool3d or Identity): Average pooling for downsampling.
        shortcut (nn.Sequential): Shortcut connection for the block.
    """
    expansion = 1

    def __init__(self, in_planes, planes, order=3, so3_size=3, stride=1, dropout=0.0, downsample_by_pooling=False, pooling_type='softmax'):
        super(ILPOBasicBlock, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = ILPO(
            in_planes, planes, kernel_size=3, order=order, so3_size=so3_size,
            stride=conv_stride, padding=1, bias=False, pooling_type=pooling_type)
        self.bn1 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(dropout, inplace=True) if dropout > 0.0 else nn.Identity()
        self.conv2 = ILPO(
            planes, planes, kernel_size=3, order=order, so3_size=so3_size,
            stride=1, padding=1, bias=False, pooling_type=pooling_type)
        self.bn2 = nn.BatchNorm3d(planes)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0) if downsample_by_pooling and stride > 1 else nn.Identity()
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=conv_stride, bias=False),
            nn.BatchNorm3d(self.expansion * planes),
        ) if stride != 1 or in_planes != self.expansion * planes else nn.Sequential()

    def forward(self, x):
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = self.bn2(self.conv2(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(out)
        return out

class ELPOBasicBlock(nn.Module):
    """
    A basic block using EquiLoPOConvolution (ELPO) for ResNet-like architectures.

    Attributes:
        conv1 (ELPO): First ELPO convolutional layer.
        bn1 (SE3BatchNorm): Batch normalization after the first ELPO layer.
        dropout (SE3Dropout or Identity): Dropout or identity function for regularization.
        conv2 (ELPO): Second ELPO convolutional layer.
        bn2 (SE3BatchNorm): Batch normalization after the second ELPO layer.
        avgpool (AvgPoolSE3 or Identity): Average pooling for downsampling.
        shortcut (nn.Sequential): Shortcut connection for the block.
        activation1, activation2 (SO3GlobalActivation or SO3LocalActivation): Activation functions.
    """
    expansion = 1

    def __init__(self, in_planes, planes, order=3, stride=1, dropout=0.0, downsample_by_pooling=False, distr_dependency=False, global_activation=False, coefficients_type='trainable'):
        super(ELPOBasicBlock, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        order_in = order if global_activation else 2 * order - 1

        self.activation1, self.activation2 = self._create_activations(
            order, planes, global_activation, distr_dependency, coefficients_type)

        self.conv1 = ELPO(
            in_planes, planes, kernel_size=3, order_in=order_in, order_out=order,
            order_filter=3, stride=conv_stride, padding=1, bias=False)
        self.bn1 = SE3BatchNorm(planes, order=order)
        self.dropout = SE3Dropout(dropout, inplace=True, num_channels=planes) if dropout > 0.0 else nn.Identity()

        self.conv2 = ELPO(
            planes, planes, kernel_size=3, order_in=order_in, order_out=order,
            order_filter=3, stride=1, padding=1, bias=False)
        self.bn2 = SE3BatchNorm(planes, order=order)
        self.avgpool = AvgPoolSE3(kernel_size=2, stride=stride, padding=0) if downsample_by_pooling and stride > 1 else nn.Identity()

        self.shortcut = self._create_shortcut(in_planes, planes, order, conv_stride)

    def _create_activations(self, order, planes, global_activation, distr_dependency, coefficients_type):
        if global_activation:
            return (SO3GlobalActivation(order, num_feat=planes),
                    SO3GlobalActivation(order, num_feat=planes))
        else:
            return (SO3LocalActivation(order, distr_dependency=distr_dependency, num_feat=planes, coefficients_type = coefficients_type),
                    SO3LocalActivation(order, distr_dependency=distr_dependency, num_feat=planes, coefficients_type = coefficients_type))

    def _create_shortcut(self, in_planes, planes, order, stride):
        if stride != 1 or in_planes != self.expansion * planes:
            return nn.Sequential(Retyper(in_planes, self.expansion * planes, order=order), SE3BatchNorm(self.expansion * planes, order=order))
        return nn.Sequential()

    def forward(self, x):
        out = self.avgpool(self.activation1(self.dropout(self.bn1(self.conv1(x)))))
        out = self.bn2(self.conv2(out))
        shortcut = self.avgpool(self.shortcut(x[:,:,:out.size(2)]))
        out += shortcut
        out = self.activation2(self.dropout(out))
        return out




class Bottleneck(nn.Module):
    """
    Bottleneck class for a deep residual network.

    Attributes:
        conv1, conv2, conv3 (nn.Conv3d): Convolutional layers.
        bn1, bn2, bn3 (nn.BatchNorm3d): Batch normalization layers.
        dropout (nn.Dropout3d or Identity): Applies dropout if dropout rate is greater than 0.
        avgpool (nn.AvgPool3d or Identity): Average pooling for downsampling.
        shortcut (nn.Sequential): Shortcut connection.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout=0.0, downsample_by_pooling=False):
        super(Bottleneck, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(dropout, inplace=True) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=conv_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0) if downsample_by_pooling and stride > 1 else nn.Identity()
        self.shortcut = self._make_shortcut(in_planes, planes, conv_stride)

    def _make_shortcut(self, in_planes, planes, stride):
        """
        Creates a shortcut connection if needed.

        Args:
            in_planes (int): Number of input planes.
            planes (int): Number of output planes.
            stride (int): Stride value.

        Returns:
            nn.Sequential: Shortcut connection layers.
        """
        if stride != 1 or in_planes != self.expansion * planes:
            return nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )
        return nn.Sequential()

    def forward(self, x):
        """
        Forward pass of the Bottleneck.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(out)
        return out


class ILPOBottleneck(nn.Module):
    """
    ILPOBottleneck class for a deep residual network using ILPO convolutions.

    Attributes:
        conv1, conv3 (nn.Conv3d): Convolutional layers.
        conv2 (ILPO): ILPO convolutional layer.
        bn1, bn2, bn3 (nn.BatchNorm3d): Batch normalization layers.
        dropout (nn.Dropout3d or Identity): Applies dropout if dropout rate is greater than 0.
        avgpool (nn.AvgPool3d or Identity): Average pooling for downsampling.
        shortcut (nn.Sequential): Shortcut connection.
    """
    expansion = 4

    def __init__(self, in_planes, planes, order=3, so3_size=3, stride=1, dropout=0.0, downsample_by_pooling=False, pooling_type='softmax'):
        super(ILPOBottleneck, self).__init__()
        conv_stride = 1 if downsample_by_pooling else stride
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(dropout, inplace=True) if dropout > 0.0 else nn.Identity()
        self.conv2 = ILPO(planes, planes, kernel_size=3, order=order, so3_size=so3_size, stride=conv_stride, padding=1, bias=False, pooling_type=pooling_type)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=stride, padding=0) if downsample_by_pooling and stride > 1 else nn.Identity()
        self.shortcut = self._make_shortcut(in_planes, planes, conv_stride)

    def _make_shortcut(self, in_planes, planes, stride):
        if stride != 1 or in_planes != self.expansion * planes:
            return nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )
        return nn.Sequential()

    def forward(self, x):
        out = self.avgpool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.avgpool(self.shortcut(x))
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    """
    General ResNet architecture for 3D inputs that supports various types of blocks.

    Attributes:
        layers (nn.Sequential): Sequential layers forming the ResNet.
        linear (nn.Linear): Final fully connected layer.
    """
    expansion = 4

    def __init__(self, block, num_blocks, order=3, so3_size=3, features=[64, 64, 128, 256, 512], 
                 strides=[1, 1, 2, 2, 2], in_channels=1, num_classes=2, dropout=0.0, 
                 downsample_by_pooling=False, pooling_type='softmax', distr_dependency=False, 
                 global_activation=False, coefficients_type='trainable'):
        super(ResNet, self).__init__()
        self.in_planes = features[0]

        # Initial convolution
        if block in [Bottleneck, BasicBlock]:
            self.initializer = lambda x: x
            self.conv1 = nn.Conv3d(in_channels, features[0], kernel_size=3, 
                                   stride=strides[0], padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(features[0])
            # Setting default activation and reducer
            self.activation = F.relu
            self.reducer = nn.Identity()
            self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        elif block in [ILPOBottleneck, ILPOBasicBlock]:
            self.initializer = lambda x: x
            self.conv1 = ILPO(in_channels, features[0], kernel_size=3, order=order, so3_size=so3_size,
                              stride=strides[0], padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(features[0])
            # Setting default activation and reducer
            self.activation = F.relu
            self.reducer = nn.Identity()
            self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        elif block == ELPOBasicBlock:
            self._setup_elop_block(features, order, strides, in_channels, distr_dependency, global_activation, coefficients_type)
        else:
            raise NotImplementedError('Block type not supported.')

        

        # Constructing layers
        self.layers = self._make_layers(block, num_blocks, features, strides, order, so3_size, dropout, downsample_by_pooling, pooling_type, distr_dependency, global_activation, coefficients_type)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        final_features = features[-1] * block.expansion if len(features) > 1 else features[-1]
        self.linear = nn.Linear(final_features, num_classes)

    def _make_layers(self, block, num_blocks, features, strides, order, so3_size, dropout, downsample_by_pooling, pooling_type, distr_dependency, global_activation, coefficients_type):
        """Constructs layers based on the block type and parameters."""
        layers = []
        for i in range(len(features) - 1):
            layers.append(self._make_layer(block, features[i + 1], num_blocks[i], order, so3_size,
                                           stride=strides[i + 1], dropout=dropout, downsample_by_pooling=downsample_by_pooling,
                                           pooling_type=pooling_type, distr_dependency=distr_dependency,
                                           global_activation=global_activation, coefficients_type=coefficients_type))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, order, so3_size, stride, dropout = 0.0, downsample_by_pooling = False, pooling_type ='softmax', distr_dependency = False, global_activation = False, poly_act = False, coefficients_type = 'trainable'):
        """Creates a single layer of blocks for the network."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i_s, stride in enumerate(strides):
            if block == Bottleneck or block == BasicBlock:
                layers.append(block(self.in_planes, planes, stride, downsample_by_pooling = downsample_by_pooling, dropout = dropout, poly_act =poly_act))
            elif block == ILPOBottleneck or block == ILPOBasicBlock:
                layers.append(block(self.in_planes, planes, order, so3_size, stride, dropout = dropout, downsample_by_pooling = downsample_by_pooling, pooling_type =pooling_type))
            elif block == ELPOBasicBlock:
                layers.append(block(self.in_planes, planes, order, stride, dropout = dropout, downsample_by_pooling = downsample_by_pooling, distr_dependency = distr_dependency, global_activation =global_activation, coefficients_type = coefficients_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _setup_elop_block(self, features, order, strides, in_channels, distr_dependency, global_activation, coefficients_type):
        """Specific setup for ELPOBasicBlock."""
        self.bn1 = SE3BatchNorm(features[0], order=order)
        self.activation = SO3GlobalActivation(order, num_feat=features[0]) if global_activation else SO3LocalActivation(order, distr_dependency=distr_dependency, num_feat=features[0], coefficients_type=coefficients_type)
        self.activation0 = SO3GlobalActivation(3, num_feat=features[0]) if global_activation else SO3LocalActivation(3, distr_dependency=distr_dependency, num_feat=features[0], coefficients_type=coefficients_type)
        order_in = 3 if global_activation else 5
        self.init_cnn = ELPO(in_channels, features[0], kernel_size=3, order_in=1, order_out = 3, order_filter = 3,
                                        stride=strides[0], padding=1, bias=False)
        
        self.conv1 = ELPO(features[0], features[0], kernel_size=3, order_in=order_in, order_out=order, order_filter=3,
                          stride=strides[0], padding=1, bias=False)
        self.reducer = SO3Softmax(2 * order - 1, distr_dependency=distr_dependency, num_feat=features[-1],
                                  global_activation=global_activation, coefficients_type=coefficients_type)
        self.bn0 = SE3BatchNorm(features[0], order=3)
        # self.dropout = SE3Dropout(0.0, inplace=True, num_channels=features[0])
        self.dropout  = nn.Identity()
        self.initializer = lambda x: self.activation0(self.dropout(self.bn0(self.init_cnn(torch.reshape(x, [x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3], x.shape[4]])))))

    def forward(self, x):
        """Forward pass of the ResNet."""
        x = self.initializer(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.reducer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


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


def ELPOResNet18(in_channels, num_classes, order = 3, dropout = 0.0, downsample_by_pooling = False, global_activation = False, coefficients_type = 'trainable'):
    return ResNet(ELPOBasicBlock, [2, 2, 2, 2], features=[4,4,4,4,4], in_channels=in_channels, num_classes=num_classes, order = order, dropout = dropout, downsample_by_pooling=downsample_by_pooling, global_activation = global_activation, coefficients_type = coefficients_type)


def ELPOResNet18Nano(in_channels, num_classes, order = 3, dropout = 0.0, downsample_by_pooling = False, global_activation = False, coefficients_type = 'trainable'):
    return ResNet(ELPOBasicBlock, [1], features=[4,4], strides=[1,1], in_channels=in_channels, num_classes=num_classes, order = order, dropout = dropout, downsample_by_pooling=downsample_by_pooling, global_activation = global_activation, coefficients_type = coefficients_type)


def ELPOResNet18Micro(in_channels, num_classes, order = 3, dropout = 0.0, downsample_by_pooling = False, global_activation = False, coefficients_type = 'trainable'):
    return ResNet(ELPOBasicBlock, [1,1], features=[4,4,4], strides=[1,1,2], in_channels=in_channels, num_classes=num_classes, order = order, dropout = dropout, downsample_by_pooling=downsample_by_pooling, global_activation = global_activation, coefficients_type = coefficients_type)



