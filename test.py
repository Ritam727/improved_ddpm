import torch

from src.attention import SpatialSelfAttention
from src.multi_head_attention import MultiHeadSpatialSelfAttention


def test_spatial_self_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    img = torch.randn(1, 64, 32, 32).to(device)
    module = SpatialSelfAttention(64, 64).to(device)
    
    with torch.no_grad():
        res = module(img)
        
        assert res.shape == img.shape, "[test_spatial_self_attention] Input image and output image dimensions different"
    print ("[test_spatial_self_attention] Passed Test")


def test_multi_head_spatial_self_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(1, 64, 32, 32).to(device)
    module = MultiHeadSpatialSelfAttention(64, 4).to(device)
    
    with torch.no_grad():
        res = module(img)
        
        assert res.shape == img.shape, "[test_multi_head_spatial_self_attention] Input image and output image dimensions different"
    print ("[test_multi_head_spatial_self_attention] Passed Test")


if __name__ == "__main__":
    test_spatial_self_attention()
    test_multi_head_spatial_self_attention()
