#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:03:06 2025

@author: luisa
"""
import numpy as np
import random
import torch

def set_deterministic(seed=0):
    """Sets all seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal (M1/M2)
    else:
        return torch.device("cpu")  # Fallback auf CPU


def add_noise(states, noise_level=0.05, seed=None):
    """
    Add Gaussian noise to each state variable (column-wise) at a relative level.
    """
    if seed is not None:
        np.random.seed(seed)

    states = np.asarray(states)

    # Ensure 2D shape: (n_states, n_time)
    if states.ndim == 1:
        states = states.reshape(1, -1)

    noisy_states = np.empty_like(states)

    for i in range(states.shape[0]):
        sigma = np.std(states[i])
        noise = np.random.normal(
            loc=0.0,
            scale=noise_level * sigma,
            size=states.shape[1]
        )
        noisy_states[i] = states[i] + noise
    print(f"\nAdding artificial noise to training states. Level: {noise_level:.4f}")
    return noisy_states
