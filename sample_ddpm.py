from torch import load, no_grad, randn, min as minimum, max as maximum

from torchvision.utils import make_grid

from src.ddpm import DDPM

from os import system
from os.path import join

from argparse import ArgumentParser

from cv2 import cvtColor, imwrite, COLOR_RGB2BGR

from yaml import safe_load, YAMLError

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
    parser.add_argument("--model_arch",
                        help = "Yaml file to load model architecture from",
                        default = "models/default.yaml")
    parser.add_argument("--num_steps",
                        help = "Number of steps for sampling",
                        type = int,
                        default = 250)
    
    args = parser.parse_args()
    
    num_images = args.num_images
    device = args.device
    prefix = args.prefix
    load_model_from = args.load_model_from
    img_shape = args.img_shape
    model_arch = args.model_arch
    num_steps = args.num_steps
    
    try:
        with open(model_arch) as stream:
            model_hyper_parameters = safe_load(stream)
    except FileNotFoundError:
        print ("Definition file not found")
    except YAMLError:
        print ("Error in YAML file")
    
    system(f"mkdir -p {prefix}")
    
    ddpm = DDPM(*model_hyper_parameters.values()).to(device)
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
        img = ddpm.sample(noise, num_steps)
        
        img -= minimum(img)
        img *= 255.0 / maximum(img)
        img = make_grid(img, nrow = 4)
        
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = cvtColor(img, COLOR_RGB2BGR)
        path = join(prefix, str(num_steps))
        imwrite(f"{path}.png", img)
