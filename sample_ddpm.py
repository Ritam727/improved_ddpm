from torch import load, no_grad, randn, min as minimum, max as maximum

from src.ddpm import DDPM

from os import system
from os.path import join

from argparse import ArgumentParser

from cv2 import cvtColor, imwrite, COLOR_RGB2BGR

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_images",
                        help = "Number of images to be generated",
                        type = int,
                        default = 4)
    parser.add_argument("--device",
                        help = "Device to use for sampling",
                        default = "cuda")
    parser.add_argument("--prefix",
                        help = "Directory in which to sample the images",
                        default = "generated_images/default")
    parser.add_argument("--load_model_from",
                        help = "File to load model weights from",
                        default = "cifar.pt")
    parser.add_argument("--img_shape",
                        help = "Shape of image to generate",
                        type = int,
                        default = 32)
    
    args = parser.parse_args()
    
    num_images = args.num_images
    device = args.device
    prefix = args.prefix
    load_model_from = args.load_model_from
    img_shape = args.img_shape
    
    system(f"mkdir -p {prefix}")
    
    ddpm = DDPM(3, 64, [1, 2, 4, 4], [2, 3], 4000, 256, 64).to(device)
    try:
        ddpm.load_state_dict(load(load_model_from, map_location = device))
    except FileNotFoundError:
        print (f"{load_model_from} does not exist, not loading from file")
    except RuntimeError:
        print (f"Key mismatch in {load_model_from}, not loading from file")
    for param in ddpm.parameters():
        param.requires_grad_(False)
    
    with no_grad():
        noise = randn(num_images, 3, img_shape, img_shape).to(device)
        for num_steps in [25, 50, 100, 200, 400, 1000, 2000]:
            img = ddpm.sample(noise, num_steps)
            
            img -= minimum(img)
            img *= 255.0 / maximum(img)
            img = img.cpu().numpy()
            
            for i, img_ in enumerate(img):
                img_ = img_.transpose(1, 2, 0)
                img_ = cvtColor(img_, COLOR_RGB2BGR)
                path = join(prefix, str(i))
                imwrite(f"{path}_{num_steps}.png", img_)
