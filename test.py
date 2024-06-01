import torch

from src.attention import SpatialSelfAttention
from src.multi_head_attention import MultiHeadSpatialSelfAttention
from src.unet_parts import TimeEmbedding, ResidualBlock, DownSample, UpSample
from src.unet import UNET
from src.diffusion import Diffusion


def test_spatial_self_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    img = torch.randn(1, 64, 32, 32).to(device)
    module = SpatialSelfAttention(64, 64).to(device)
    
    with torch.no_grad():
        res = module(img)
        
        assert res.shape == img.shape, "[test_spatial_self_attention] Input image and output image dimensions do not match"
    print ("[test_spatial_self_attention] Passed Test")


def test_multi_head_spatial_self_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(1, 64, 32, 32).to(device)
    module = MultiHeadSpatialSelfAttention(64, 4).to(device)
    
    with torch.no_grad():
        res = module(img)
        
        assert res.shape == img.shape, "[test_multi_head_spatial_self_attention] Input image and output image dimensions do not match"
    print ("[test_multi_head_spatial_self_attention] Passed Test")


def test_time_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    t = torch.randint(0, 100, (4,)).to(device)
    module = TimeEmbedding(100, 64, 16).to(device)
    
    with torch.no_grad():
        res = module(t)
        
        assert len(res.shape) == 2, "[test_time_embedding] Output dimensions should be in 2 axes"
        assert res.shape == torch.Size([t.shape[0], 16]), "[test_time_embedding] Output dimensions should be equal to (B, d_time)"
    print ("[test_time_embedding] Passed Test")


def test_residual_block():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(2, 64, 32, 32).to(device)
    t = torch.randn(2, 16).to(device)
    module = ResidualBlock(64, 32, 16).to(device)
    
    with torch.no_grad():
        res = module(img, t)
        
        assert res.shape[1] == 32, "[test_residual_block] Output channels do not match"
        assert res.shape[2:] == img.shape[2:], "[test_residual_block] Output image shape does not match input image shape"
    print ("[test_residual_block] Passed Test")


def test_downsample():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(2, 64, 32, 32).to(device)
    module = UpSample(64, 64).to(device)
    
    with torch.no_grad():
        res = module(img)
        
        assert res.shape[1] == 64, "[test_residual_block] Output channels do not match"
        H, W = img.shape[2:]
        assert res.shape[2:] == torch.Size([H * 2, W * 2]), "[test_downsample] Output image shape does not match input image shape"
    print ("[test_downsample] Passed Test")


def test_upsample():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(2, 64, 32, 32).to(device)
    module = DownSample(64, 64).to(device)
    
    with torch.no_grad():
        res = module(img)
        
        assert res.shape[1] == 64, "[test_residual_block] Output channels do not match"
        H, W = img.shape[2:]
        assert res.shape[2:] == torch.Size([H // 2, W // 2]), "[test_upsample] Output image shape does not match input image shape"
    print ("[test_upsample] Passed Test")


def test_unet():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    module = UNET(3, 32, [1, 2, 4, 4], [2, 3], 100, 64, 16).to(device)
    
    with torch.no_grad():
        res = module(img, t)
        
        assert res.shape[1] == 2 * img.shape[1], "[test_unet] Output channels must be twice that of input channels"
        assert res.shape[2:] == img.shape[2:], "[test_unet] Input image dimensions and output image dimensions must be same"
    print ("[test_unet] Passed Test")


def test_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    img = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    noise = torch.randn(2, 3, 32, 32).to(device)
    module = Diffusion(100).to(device)
    
    with torch.no_grad():
        res = module(img, noise, t)
        
        assert res.shape == img.shape, "[test_diffusion] Input image and output image shape must be same"
    print ("[test_diffusion] Passed Test")


if __name__ == "__main__":
    test_spatial_self_attention()
    test_multi_head_spatial_self_attention()
    test_time_embedding()
    test_residual_block()
    test_downsample()
    test_upsample()
    test_unet()
    test_diffusion()
