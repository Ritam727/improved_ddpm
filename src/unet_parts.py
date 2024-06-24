import torch
from torch import nn
from torch.nn import functional as F

import math


class TimeEmbedding(nn.Module):
    
    def __init__(self, time_dim : int, d_model : int, d_time : int):
        """
            Computes embedding for timestep
        """
        
        super().__init__()
        positional_encodings = torch.zeros(time_dim, d_time)
        positions = torch.arange(0, time_dim).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_time, 2).float() * (-math.log(10000) / d_time))
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
        self.temb = nn.Sequential(
            nn.Embedding.from_pretrained(positional_encodings),
            nn.Linear(d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_time)
        )
        
    def forward(self, t : torch.LongTensor) -> torch.Tensor:
        """
            Input Shape : (B)
            Output Shape : (B, D)
        """
        
        return self.temb(t)


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, d_time : int, batch_norm : bool = False):
        """
            Residual Block computing time embedding and convolutions on the image
        """
        
        super().__init__()
        self.time_encoding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_time, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.group_norm = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
            
    
    def forward(self, x : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
            Input Shape : (B, C_in, H, W), (B, D)
            Output Shape : (B, C_out, H, W)
        """
        
        t = self.time_encoding_layer(t)
        
        residue = x
        x = self.block1(x) + t.unsqueeze(-1).unsqueeze(-1)
        x = self.block2(x)
        x = self.group_norm(x)
        
        return x + self.residual_layer(residue)


class DownSample(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int):
        """
            Halves the dimensions of the input image using a convolutional layer
        """
        
        super().__init__()
        self.layer = nn.Sequential(
            nn.SiLU(),
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
            Input Shape : (B, C_in, H, W)
            Output Shape : (B, C_out, H // 2, W // 2)
        """
        
        return self.layer(x)


class UpSample(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int):
        """
            Doubles the dimensions of input image and then passes through a convolutional layer
        """
        
        super().__init__()
        self.layer = nn.Sequential(
            nn.SiLU(),
            nn.GroupNorm(32, in_channels),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
            Input Shape : (B, C_in, H, W),
            Output Shape : (B, C_out, H * , W * 2)
        """
        
        return self.layer(x)


class SwitchSequential(nn.Sequential):
    
    def forward(self, x : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
            Input Shape : (B, C_in, H, W), (B, D)
            Output Shape : (B, C_out, H, W)
        """
        
        for layer in self:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        return x
