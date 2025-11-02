import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'tanh':
        bottleneck = TanhBottleneck()
    elif bottleneck_type == 'vae':
        bottleneck = VAEBottleneck()
    elif bottleneck_type == 'l2_norm':
        bottleneck = L2Bottleneck()
    elif bottleneck_type == "wasserstein":
        bottleneck = WassersteinBottleneck(**bottleneck_config.get("config", {}))
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')
    
    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck

class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()
        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

class TanhBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)
        self.tanh = nn.Tanh()

    def encode(self, x, return_info=False):
        info = {}
        x = torch.tanh(x)
        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl
        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def compute_mean_kernel(x, y):
    kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
    return torch.exp(-kernel_input).mean()

def compute_mmd(latents):
    latents_reshaped = latents.permute(0, 2, 1).reshape(-1, latents.shape[1])
    noise = torch.randn_like(latents_reshaped)
    latents_kernel = compute_mean_kernel(latents_reshaped, latents_reshaped)
    noise_kernel = compute_mean_kernel(noise, noise)
    latents_noise_kernel = compute_mean_kernel(latents_reshaped, noise)
    mmd = latents_kernel + noise_kernel - 2 * latents_noise_kernel
    return mmd.mean()

class WassersteinBottleneck(Bottleneck):
    def __init__(self, noise_augment_dim: int = 0, bypass_mmd: bool = False):
        super().__init__(is_discrete=False)
        self.noise_augment_dim = noise_augment_dim
        self.bypass_mmd = bypass_mmd
    
    def encode(self, x, return_info=False):
        info = {}
        if self.training and return_info:
            if self.bypass_mmd:
                mmd = torch.tensor(0.0)
            else:
                mmd = compute_mmd(x)
            info["mmd"] = mmd
        if return_info:
            return x, info
        return x

    def decode(self, x):
        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)
        return x

class L2Bottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)
    
    def encode(self, x, return_info=False):
        info = {}
        x = F.normalize(x, dim=1)
        if return_info:
            return x, info
        else:
            return x
    
    def decode(self, x):
        return F.normalize(x, dim=1)