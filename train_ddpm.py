from torch import load, save, randn, randint, exp, log
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.ddpm import DDPM

from argparse import ArgumentParser

from torchvision.datasets import CIFAR10, CelebA
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, Normalize

from tqdm import tqdm

from yaml import safe_load, YAMLError


def train(module : DDPM, dataset : Dataset, epochs : int, batch_size : int, learning_rate : float, device : str, save_model_as, progress : bool = True):
    loader = DataLoader(dataset, shuffle = True, batch_size = batch_size)
    optim = AdamW(module.unet.parameters(), lr = learning_rate)
    
    for epoch in range(epochs):
        if not progress:
            print (f"Running epoch {epoch + 1}")
        
        itr = tqdm(loader, unit = "batch") if progress else loader
        for img, _ in itr:
            if isinstance(itr, tqdm):
                itr.set_description(f"Training {epoch + 1}/{epochs}")
            
            optim.zero_grad(set_to_none = True)
            
            img = img.to(device)
            noise = randn(img.shape).to(device)
            t = randint(0, module.time_steps, (img.shape[0],)).long().to(device)
            x_t, eps, v = module(img, noise, t)
            
            sqrt_alpha_t = module.diffusion.sqrt_alpha[t].view(img.shape[0], 1, 1, 1)
            beta_t = module.diffusion.beta[t].view(img.shape[0], 1, 1, 1)
            beta_bar_t = module.diffusion.beta_bar[t].view(img.shape[0], 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = module.diffusion.sqrt_one_minus_alpha_bar[t + 1].view(img.shape[0], 1, 1, 1)
            sqrt_alpha_bar_t_minus_one = module.diffusion.sqrt_alpha_bar[t].view(img.shape[0], 1, 1, 1)
            one_minus_alpha_bar_t_minus_one = module.diffusion.sqrt_one_minus_alpha_bar[t].square().view(img.shape[0], 1, 1, 1)
            one_minus_alpha_bar_t = module.diffusion.sqrt_one_minus_alpha_bar[t + 1].square().view(img.shape[0], 1, 1, 1)
            
            u = 1 / sqrt_alpha_t * (img - beta_t / sqrt_one_minus_alpha_bar_t * eps)
            u_bar = sqrt_alpha_bar_t_minus_one * beta_t / one_minus_alpha_bar_t * img + sqrt_alpha_t * one_minus_alpha_bar_t_minus_one / one_minus_alpha_bar_t * x_t
            log_sigma = v * log(beta_t) + (1.0 - v) * log(beta_bar_t)
            log_beta_bar_t = log(beta_bar_t)
            
            mse = (eps - noise).square().mean()
            kl = 0.5 * (-1 + log_sigma - log_beta_bar_t + exp(log_beta_bar_t - log_sigma) + (u.detach() - u_bar).square() * exp(-log_sigma)).mean()
            loss = mse + 0.001 * kl
            
            loss.backward()
            optim.step()
            
            if isinstance(itr, tqdm):
                itr.set_postfix(mse = mse.item(), kl = kl.item(), loss = loss.item())
        save(module.state_dict(), save_model_as)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset",
                        help = "Name of dataset to train the model on",
                        default = "cifar10")
    parser.add_argument("--save_model_as",
                        help = "Name of file to save model weights to",
                        default = "cifar.pt")
    parser.add_argument("--device",
                        help = "Device to use for training",
                        default = "cuda")
    parser.add_argument("--epochs",
                        help = "Number of epochs to train model",
                        type = int,
                        default = 20)
    parser.add_argument("--batch_size",
                        help = "Batch size to use during training",
                        type = int,
                        default = 32)
    parser.add_argument("--learning_rate",
                        help = "Learning rate",
                        type = float,
                        default = 1e-5)
    parser.add_argument("--progress",
                        help = "Option for showing progress bar",
                        type = int,
                        default = True)
    parser.add_argument("--model_arch",
                        help = "Yaml file to load model architecture from",
                        default = "models/default.yaml")
    
    args = parser.parse_args()
    
    dataset_name = str(args.dataset)
    save_model_as = str(args.save_model_as)
    device = str(args.device)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    progress = int(args.progress)
    model_arch = str(args.model_arch)
    
    try:
        with open(model_arch) as stream:
            model_hyper_parameters = safe_load(stream)
    except FileNotFoundError:
        print ("Definition file not found")
    except YAMLError:
        print ("Error in YAML file")
    
    if dataset_name.upper() == "CIFAR10":
        dataset = CIFAR10("data", train = True, transform = Compose([ToTensor(), RandomHorizontalFlip(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), download = True)
    elif dataset_name.upper() == "CELEBA":
        dataset = CelebA("data", "train", target_type = "identity", transform = Compose([ToTensor(), RandomHorizontalFlip(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), download = True)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")
    
    ddpm = DDPM(*model_hyper_parameters.values()).to(device)
    try:
        ddpm.load_state_dict(load(save_model_as, map_location = device))
    except FileNotFoundError:
        print (f"{save_model_as} does not exist, not loading from file")
    except RuntimeError:
        print (f"Key mismatch in {save_model_as}, not loading from file")
    
    train(ddpm, dataset, epochs, batch_size, learning_rate, device, save_model_as, progress)
