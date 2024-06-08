import torch
from torch import nn
from torch.nn import functional as F

from .diffusion import Diffusion
from .unet import UNET

import tqdm


class DDPM(nn.Module):
    
    def __init__(self, in_channels : int, ch_init : int, ch_mult : list[int], attn_layers : list[int], time_dim : int, d_model : int, d_time : int):
        """
            Computes forward process noisy image and returns predicted noise as well as log variance from backward process
        """
        
        super().__init__()
        self.time_steps = time_dim
        self.diffusion = Diffusion(time_dim)
        self.unet = UNET(in_channels, ch_init, ch_mult, attn_layers, time_dim, d_model, d_time)
        
    def forward(self, x : torch.Tensor, noise : torch.Tensor, t : torch.LongTensor) -> torch.Tensor:
        """
            Input Shape : (B, C, H, W), (B, C, H, W), (B)
            Output Shape : (B, 2 * C, H, W)
        """
        
        assert x.shape == noise.shape, "Image input and noise input shape must be same"
        
        x_t = self.diffusion(x, noise, t)
        eps, v = torch.chunk(self.unet(x_t, t), 2, dim = 1)
        
        return x_t, eps, F.sigmoid(v)
    
    def sample(self, x : torch.Tensor, num_steps : int) -> torch.Tensor:
        """
            Input Shape : (B, C, H, W)
            Output Shape : (B, C, H, W)
        """
        
        b, c, h, w = x.shape
        
        indices = torch.linspace(0, self.time_steps - 1, num_steps).int()
        alpha_bar = self.diffusion.alpha_bar[indices].squeeze()
        sqrt_one_minus_alpha_bar = self.diffusion.sqrt_one_minus_alpha_bar[indices].squeeze()
        beta = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).squeeze()
        sqrt_alpha = (1.0 - beta).sqrt()
        beta_bar = (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]).squeeze() * beta
        
        for t in tqdm.trange(num_steps - 1, 0, -1):
            z = torch.randn(b, c, h, w).to(x.device) if t > 1 else torch.zeros(b, c, h, w).to(x.device)
            time_tensor = torch.tensor([t - 1] * b).long().to(x.device)
            eps, v = torch.chunk(self.unet(x, time_tensor), 2, dim = 1)
            
            sqrt_alpha_t = sqrt_alpha[t - 1]
            beta_t = beta[t - 1]
            beta_bar_t = beta_bar[t - 1]
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t]
            
            log_var = v * torch.log(beta_t) + (1.0 - v) * torch.log(beta_bar_t)
            
            assert torch.isnan(eps).sum() == 0, f"Encountered NaN values during sampling at time step {num_steps - t} in module output"
            assert torch.isnan(log_var).sum() == 0, f"Encountered NaN values during sampling at time step {num_steps - t} while log operation"
            
            x = 1 / sqrt_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * eps) + torch.exp(0.5 * log_var) * z
            
            
        return x
