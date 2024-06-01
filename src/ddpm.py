import torch
from torch import nn

from .diffusion import Diffusion
from .unet import UNET


class DDPM(nn.Module):
    
    def __init__(self, in_channels : int, ch_init : int, ch_mult : list[int], attn_layers : list[int], time_dim : int, d_model : int, d_time : int):
        """
            Computes forward process noisy image and returns predicted noise as well as log variance from backward process
        """
        
        super().__init__()
        self.diffusion = Diffusion(time_dim)
        self.unet = UNET(in_channels, ch_init, ch_mult, attn_layers, time_dim, d_model, d_time)
        
    def forward(self, x : torch.Tensor, noise : torch.Tensor, t : torch.LongTensor) -> torch.Tensor:
        """
            Input Shape : (B, C, H, W), (B, C, H, W), (B)
            Output Shape : (B, 2 * C, H, W)
        """
        
        assert x.shape == noise.shape, "Image input and noise input shape must be same"
        
        x_t = self.diffusion(x, noise, t)
        mu, log_var = torch.chunk(self.unet(x_t, t), 2, dim = 1)
        
        return mu, log_var