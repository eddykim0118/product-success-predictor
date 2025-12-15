"""
3D ResNet
=========

3D ResNet architectures for volumetric classification.
Based on "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
"""

from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution."""
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    """Basic ResNet block."""
    
    expansion = 1
    
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block."""
    
    expansion = 4
    
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet for volumetric classification.
    """
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 14,
        zero_init_residual: bool = False,
        base_width: int = 64,
        dropout: float = 0.5,
    ):
        """
        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: Number of blocks per layer
            in_channels: Input channels (1 for CT)
            num_classes: Number of output classes
            zero_init_residual: Zero-init last BN in each residual branch
            base_width: Base width of network
            dropout: Dropout rate before final FC
        """
        super().__init__()
        
        self.in_planes = base_width
        self.base_width = base_width
        
        # Stem
        self.conv1 = nn.Conv3d(
            in_channels, self.in_planes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)
        
        # Weight initialization
        self._init_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        x = self.forward_features(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def resnet10_3d(in_channels: int = 1, num_classes: int = 14, **kwargs) -> ResNet3D:
    """ResNet-10 3D."""
    return ResNet3D(BasicBlock, [1, 1, 1, 1], in_channels, num_classes, **kwargs)


def resnet18_3d(in_channels: int = 1, num_classes: int = 14, **kwargs) -> ResNet3D:
    """ResNet-18 3D."""
    return ResNet3D(BasicBlock, [2, 2, 2, 2], in_channels, num_classes, **kwargs)


def resnet34_3d(in_channels: int = 1, num_classes: int = 14, **kwargs) -> ResNet3D:
    """ResNet-34 3D."""
    return ResNet3D(BasicBlock, [3, 4, 6, 3], in_channels, num_classes, **kwargs)


def resnet50_3d(in_channels: int = 1, num_classes: int = 14, **kwargs) -> ResNet3D:
    """ResNet-50 3D."""
    return ResNet3D(Bottleneck, [3, 4, 6, 3], in_channels, num_classes, **kwargs)


def resnet101_3d(in_channels: int = 1, num_classes: int = 14, **kwargs) -> ResNet3D:
    """ResNet-101 3D."""
    return ResNet3D(Bottleneck, [3, 4, 23, 3], in_channels, num_classes, **kwargs)


def resnet152_3d(in_channels: int = 1, num_classes: int = 14, **kwargs) -> ResNet3D:
    """ResNet-152 3D."""
    return ResNet3D(Bottleneck, [3, 8, 36, 3], in_channels, num_classes, **kwargs)


class ResNet3DWithDropout(ResNet3D):
    """
    ResNet3D with additional dropout layers between blocks.
    """
    
    def __init__(self, *args, block_dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_dropout = nn.Dropout3d(block_dropout)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.block_dropout(x)
        x = self.layer2(x)
        x = self.block_dropout(x)
        x = self.layer3(x)
        x = self.block_dropout(x)
        x = self.layer4(x)
        
        return x

