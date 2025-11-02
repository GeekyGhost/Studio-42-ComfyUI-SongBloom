import torch
import math
from tqdm import trange, tqdm

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

@torch.no_grad()
def sample_discrete_euler_with_temperature(model, x, steps, temperature=1.0, sigma_max=1.0, prog_bar=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])
    noise = x

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)
    # all = {}
    x = torch.zeros_like(noise)
    if temperature >= sigma_max:
        x = noise
        
    #alphas, sigmas = 1-t, t
    iterator = tqdm(zip(t[:-1], t[1:]), total=steps) if prog_bar else zip(t[:-1], t[1:])
    for t_curr, t_prev in iterator:
            # Broadcast the current timestep to the correct shape
                
            t_curr_tensor = t_curr * torch.ones(
                (x.shape[0],), dtype=x.dtype, device=x.device
            )
            dt = t_prev - t_curr  # we solve backwards in our formulation
            v = model(x, t_curr_tensor, **extra_args)
            # all[t_curr.item()] = x-t_curr*v
            if  t_curr > temperature and t_prev <= temperature:
                x_0 = x - v
                x = (1-t_prev) * x_0 + t_prev * noise
            else:
                x = x + dt * v #.denoise(x, denoiser, t_curr_tensor, cond, uc)
            
    # If we are on the last timestep, output the denoised image
    return x #, all

import numpy as np

@torch.no_grad()
def sample_discrete_euler_spiral(model, x, steps, temperature=1.0, sigma_max=1.0, 
                                 num_paths=2, spiral_strength=0.1, prog_bar=False, 
                                 combination_method='weighted_avg', **extra_args):
    """
    Multi-path spiral sampling using Euler method.
    Each path follows a slightly different spiral trajectory through noise space.
    
    Args:
        model: The diffusion model
        x: Starting noise
        steps: Number of denoising steps
        temperature: Temperature parameter from original sampler
        sigma_max: Maximum noise level
        num_paths: Number of spiral paths to sample
        spiral_strength: How much the spiral affects the trajectory (0.0 = no spiral, 1.0 = strong spiral)
        prog_bar: Show progress bar
        combination_method: 'weighted_avg', 'best_path', or 'ensemble'
        **extra_args: Additional arguments for the model
    
    Returns:
        Combined result from all spiral paths
    """
    
    def create_spiral_schedule(steps, sigma_max, phase_offset=0, spiral_freq=2):
        """Create a spiral-modulated time schedule"""
        # Base linear schedule
        t_linear = torch.linspace(sigma_max, 0, steps + 1)
        
        # Create spiral modulation
        theta = torch.linspace(0, spiral_freq * 2 * np.pi, steps + 1) + phase_offset
        spiral_mod = spiral_strength * torch.sin(theta) * t_linear / sigma_max
        
        # Apply spiral modulation (keep within bounds)
        t_spiral = t_linear + spiral_mod
        t_spiral = torch.clamp(t_spiral, 0, sigma_max)
        
        return t_spiral
    
    def sample_single_path(noise, t_schedule, path_id):
        """Sample a single spiral path"""
        ts = noise.new_ones([noise.shape[0]])
        x_path = torch.zeros_like(noise)
        
        if temperature >= sigma_max:
            x_path = noise.clone()
        
        path_results = []
        iterator = tqdm(zip(t_schedule[:-1], t_schedule[1:]), 
                       total=steps, desc=f"Path {path_id+1}") if prog_bar else zip(t_schedule[:-1], t_schedule[1:])
        
        for t_curr, t_prev in iterator:
            # Broadcast the current timestep
            t_curr_tensor = t_curr * torch.ones(
                (x_path.shape[0],), dtype=x_path.dtype, device=x_path.device
            )
            dt = t_prev - t_curr
            
            # Get model prediction
            v = model(x_path, t_curr_tensor, **extra_args)
            
            # Apply temperature switching logic from original
            if t_curr > temperature and t_prev <= temperature:
                x_0 = x_path - v
                x_path = (1-t_prev) * x_0 + t_prev * noise
            else:
                x_path = x_path + dt * v
                
            # Store intermediate results for potential analysis
            if len(path_results) < 5:  # Store first few steps
                path_results.append(x_path.clone())
        
        return x_path, path_results
    
    # Generate different spiral paths
    all_paths = []
    all_intermediates = []
    
    for path_idx in range(num_paths):
        # Create unique phase offset for each path
        phase_offset = path_idx * (2 * np.pi / num_paths)
        
        # Create spiral schedule for this path
        t_schedule = create_spiral_schedule(steps, sigma_max, phase_offset)
        
        # Sample this path
        path_result, intermediates = sample_single_path(x, t_schedule, path_idx)
        all_paths.append(path_result)
        all_intermediates.append(intermediates)
    
    # Combine paths based on chosen method
    if combination_method == 'weighted_avg':
        # Simple average of all paths
        final_result = torch.stack(all_paths).mean(dim=0)
        
    elif combination_method == 'best_path':
        # Select path with lowest variance (most stable)
        path_variances = [torch.var(path).item() for path in all_paths]
        best_idx = np.argmin(path_variances)
        final_result = all_paths[best_idx]
        
    elif combination_method == 'ensemble':
        # Weighted combination based on path quality metrics
        weights = []
        for path in all_paths:
            # Simple quality metric: negative variance (lower variance = higher weight)
            quality = 1.0 / (1.0 + torch.var(path).item())
            weights.append(quality)
        
        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        
        # Weighted combination
        final_result = torch.zeros_like(all_paths[0])
        for i, (path, weight) in enumerate(zip(all_paths, weights)):
            final_result += weight * path
    
    else:
        # Fallback to simple average
        final_result = torch.stack(all_paths).mean(dim=0)
    
    return final_result

@torch.no_grad()
def sample_discrete_euler_with_temperature_pingpong(model, x, steps, temperature=1.0, sigma_max=1.0, 
                                                      ping_pong_amplitude=0.1, prog_bar=False, **extra_args):
    """Alternative ping pong implementation with amplitude-based oscillation"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])
    noise = x

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)
    x = torch.zeros_like(noise)
    if temperature >= sigma_max:
        x = noise
        
    iterator = tqdm(zip(t[:-1], t[1:]), total=steps) if prog_bar else zip(t[:-1], t[1:])
    
    for step_idx, (t_curr, t_prev) in enumerate(iterator):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * torch.ones(
            (x.shape[0],), dtype=x.dtype, device=x.device
        )
        dt = t_prev - t_curr  # we solve backwards in our formulation
        
        # Get velocity at current point
        v = model(x, t_curr_tensor, **extra_args)
        
        # Ping pong: add oscillation to the step
        ping_pong_factor = ping_pong_amplitude * torch.sin(torch.tensor(step_idx * 2 * 3.14159 / 4))
        effective_dt = dt * (1 + ping_pong_factor)
        
        # Take the step
        x_new = x + effective_dt * v
        
        # Optional: Add a corrector step for better stability
        if ping_pong_amplitude > 0:
            t_new_tensor = t_prev * torch.ones(
                (x.shape[0],), dtype=x.dtype, device=x.device
            )
            v_corrector = model(x_new, t_new_tensor, **extra_args)
            x = x + 0.5 * dt * (v + v_corrector)  # Heun's method style correction
        else:
            x = x_new
        
        # Temperature-based noise injection
        if t_curr > temperature and t_prev <= temperature:
            x_0 = x - v
            x = (1 - t_prev) * x_0 + t_prev * noise
            
    return x