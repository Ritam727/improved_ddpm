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
        positional_encodings = torch.zeros(time_dim, d_model)
        positions = torch.arange(0, time_dim).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
        self.temb = nn.Sequential(
            nn.Embedding.from_pretrained(positional_encodings),
            nn.Linear(d_model, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time)
        )
        
    def forward(self, t : torch.LongTensor) -> torch.Tensor:
        """
            Input Shape : (B)
            Output Shape : (B, D)
        """
        
        return self.temb(t)
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, d_time : int):
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
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
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
        
        return x + self.residual_layer(residue)