"""
3D U-Net
========

3D U-Net architecture for aneurysm segmentation.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    Double convolution block: (Conv3D -> BN -> ReLU) * 2
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        trilinear: bool = True,
    ):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.
    
    Architecture:
        Encoder: 4 down blocks with max pooling
        Bottleneck: Double conv at lowest resolution
        Decoder: 4 up blocks with skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 32,
        trilinear: bool = True,
        deep_supervision: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            base_filters: Number of filters in first layer
            trilinear: Use trilinear upsampling vs transposed conv
            deep_supervision: Enable deep supervision outputs
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        
        factor = 2 if trilinear else 1
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, trilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, trilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, trilinear)
        self.up4 = Up(base_filters * 2, base_filters, trilinear)
        
        # Output
        self.outc = OutConv(base_filters, out_channels)
        
        # Deep supervision heads (optional)
        if deep_supervision:
            self.ds1 = OutConv(base_filters * 8 // factor, out_channels)
            self.ds2 = OutConv(base_filters * 4 // factor, out_channels)
            self.ds3 = OutConv(base_filters * 2 // factor, out_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Segmentation mask of shape (B, out_channels, D, H, W)
            If deep_supervision: tuple of (main_output, [ds_outputs])
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        
        # Output
        out = self.outc(d1)
        
        if self.deep_supervision and self.training:
            ds1 = F.interpolate(self.ds1(d4), size=out.shape[2:], mode='trilinear', align_corners=True)
            ds2 = F.interpolate(self.ds2(d3), size=out.shape[2:], mode='trilinear', align_corners=True)
            ds3 = F.interpolate(self.ds3(d2), size=out.shape[2:], mode='trilinear', align_corners=True)
            return out, [ds1, ds2, ds3]
        
        return out
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary segmentation prediction.
        
        Args:
            x: Input tensor
            threshold: Threshold for binary prediction
            
        Returns:
            Binary mask
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()


class AttentionUNet3D(UNet3D):
    """
    3D U-Net with attention gates for improved skip connections.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        base_filters = kwargs.get('base_filters', 32)
        
        # Attention gates
        self.att1 = AttentionGate3D(base_filters * 8, base_filters * 8)
        self.att2 = AttentionGate3D(base_filters * 4, base_filters * 4)
        self.att3 = AttentionGate3D(base_filters * 2, base_filters * 2)
        self.att4 = AttentionGate3D(base_filters, base_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        x4 = self.att1(x4, x5)
        d4 = self.up1(x5, x4)
        
        x3 = self.att2(x3, d4)
        d3 = self.up2(d4, x3)
        
        x2 = self.att3(x2, d3)
        d2 = self.up3(d3, x2)
        
        x1 = self.att4(x1, d2)
        d1 = self.up4(d2, x1)
        
        return self.outc(d1)


class AttentionGate3D(nn.Module):
    """
    3D Attention gate for U-Net skip connections.
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: Optional[int] = None):
        super().__init__()
        
        if F_int is None:
            F_int = F_l // 2
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Skip connection from encoder
            g: Gating signal from decoder
        """
        # Upsample g to match x size
        g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=True)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

