#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:16:37 2025

@author: peterson
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
            "ConvDecoder",
            "train_decoder",
          ]

import opinf
device = opinf.utils.get_device()
Params = opinf.parameters.Params()  # call parameters from dataclass


class ConvDecoder(nn.Module):
    """
    Convolution-based decoder with flexible normalization, dropout, and activation.
    Designed for POD-CNN reconstruction tasks.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims=[64, 512],
        conv_channels=[32, 16, 8, 2],
        kernel_sizes=[3, 3, 3, 5],
        strides=[2, 2, 2, 2],
        paddings=[1, 1, 1, 2],
        dropout_linear=0.0,
        dropout_conv=0.0,
        norm_type: str = "batch",   # "batch", "layer", or "none"
        activation_fn: str = "silu",  # Ersetze LeakyReLU durch SiLU (auch bekannt als Swish) oder ELU
    ):
        super().__init__()
        self.norm_type = norm_type.lower()
        self.dropout_linear = dropout_linear
        self.dropout_conv = dropout_conv
        self.activation = self._get_activation(activation_fn)

        # === Linear (fully connected) network ===
        print(self.activation)
        layers = []
        in_features = 2 * latent_dim
        for out_features in hidden_dims:
            block = [nn.Linear(in_features, out_features)]
            if self.norm_type == "batch":
                block.append(nn.BatchNorm1d(out_features))
            elif self.norm_type == "layer":
                block.append(nn.LayerNorm(out_features))
            block.append(self.activation)
            if dropout_linear > 0:
                block.append(nn.Dropout(dropout_linear))
            layers.extend(block)
            in_features = out_features
        self.linear_net = nn.Sequential(*layers)

        # Prepare for reshaping into (channels, spatial)
        self.initial_channels = hidden_dims[0]
        self.initial_spatial = hidden_dims[-1] // self.initial_channels

        # === Convolutional transpose network ===
        conv_blocks = []
        in_channels = self.initial_channels
        for i, out_channels in enumerate(conv_channels):
            conv = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                output_padding=1,
            )
            block = [conv]
            if self.norm_type == "batch":
                block.append(nn.BatchNorm1d(out_channels))
            elif self.norm_type == "layer":
                block.append(nn.GroupNorm(1, out_channels))  # instance-like norm
            if i < len(conv_channels) - 1:
                block.append(self.activation)
                if dropout_conv > 0:
                    block.append(nn.Dropout(dropout_conv))
            conv_blocks.append(nn.Sequential(*block))
            in_channels = out_channels
        self.conv_net = nn.ModuleList(conv_blocks)

        # === Final projection layer ===
        final_size = self._calculate_output_size()
        self.final_proj = nn.Linear(final_size, output_dim)

        # Optional upsampling before conv layers
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

        # Initialize weights
        self.apply(self._init_weights)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.dropout_linear = dropout_linear
        self.dropout_conv = dropout_conv
        self.norm_type = norm_type
        self.activation_fn_name = activation_fn

    def _get_activation(self, name: str):
            """Return activation module based on string name."""
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            elif name == "leakyrelu":
                return nn.LeakyReLU(0.01)
            elif name == "elu":
                return nn.ELU()
            elif name == "selu":
                return nn.SELU()
            elif name == "silu" or name == "swish":
                return nn.SiLU()
            elif name == "gelu":
                return nn.GELU()
            elif name == "softplus":
                return nn.Softplus()
            else:
                raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self, m):
        """Custom weight initialization."""
        if isinstance(m, (nn.Linear, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _calculate_output_size(self):
        """Compute flattened size after all conv-transpose layers."""
        size = self.initial_spatial
        size = int(size * 2)  # Upsample
        for conv in self.conv_net:
            c = conv[0]  # ConvTranspose1d is always first in the block
            size = (size - 1) * c.stride[0] + c.kernel_size[0] - 2 * c.padding[0] + c.output_padding[0]
        return size * self.conv_net[-1][0].out_channels

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass from latent vector z to reconstructed signal."""
        # Augment latent vector with squared terms
        z2 = z ** 2
        z_all = torch.cat((z, z2), dim=-1)

        # Dense network
        h = self.linear_net(z_all)

        # Reshape into (batch, channels, spatial)
        channels = self.initial_channels
        spatial = h.shape[1] // channels
        h = h.view(h.size(0), channels, spatial)
        h = self.upsample(h)

        # Convolutional transpose network
        for block in self.conv_net:
            h = block(h)

        # Flatten and project to output_dim
        h = h.flatten(1)
        return self.final_proj(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)


def smoothness_loss(x_hat: torch.Tensor) -> torch.Tensor:
    # Penalize differences between adjacent elements
    return torch.mean((x_hat[:, 1:] - x_hat[:, :-1]).pow(2))


def train_decoder(train_dl, val_dl, basis, latent_dim, seed=None):
    """
    Train the ConvDecoder model with given dataloaders and basis.

    Parameters
    ----------
    train_dl : torch.utils.data.DataLoader
        Training dataloader (inputs already scaled & augmented).
    val_dl   : torch.utils.data.DataLoader
        Validation dataloader.
    basis    : np.ndarray
        POD basis, shape (output_dim, ...).
    latent_dim : int
        Dimension of latent space (r_T + r_T).
    """
    current_seed = seed if seed is not None else 0
    opinf.utils.set_deterministic(seed=current_seed)

    # --- Build ConvDecoder with parameters from Params ---
    decoder = ConvDecoder(
        latent_dim=latent_dim,
        output_dim=basis.shape[0],
        hidden_dims=[64, 512],
        conv_channels=Params.CNN_conv_channel,
        kernel_sizes=[3, 3, 3, 5],
        strides=[2, 2, 2, 2],
        paddings=[1, 1, 1, 2],
        dropout_linear=Params.CNN_dropout_rate_lin,
        dropout_conv=Params.CNN_dropout_rate_conv,
        norm_type="batch",
        activation_fn=Params.CNN_activation,
    ).float().to(device)

    # --- Optimizer & scheduler ---
    optim = torch.optim.NAdam(
        decoder.parameters(),
        lr=Params.CNN_learn_rate,
        weight_decay=Params.CNN_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=1000, gamma=0.1
    )

    # --- Loss functions ---
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    basis_tensor = torch.from_numpy(basis).float().to(device)

    if Params.output:
        print("CNN Decoder training started")

    # --- POD baseline (reference loss on val set) ---
    pod_losses = []
    with torch.no_grad():
        for input_val, target_val in val_dl:
            input_val = input_val.float().to(device)
            target_val = target_val.float().to(device)
            pod_pred = input_val @ basis_tensor.T
            pod_losses.append(mse_loss(pod_pred, target_val).item())
    pod_loss_ref = float(np.mean(pod_losses)) if len(pod_losses) > 0 else float("inf")

    # --- Training loop setup ---
    err_t, err_val = [], []
    best_val_loss = float("inf")
    best_model = deepcopy(decoder.state_dict())
    counter = 0

    # --- Training epochs ---
    for epoch in range(Params.CNN_epochs):
        decoder.train()
        epoch_loss, n_batches = 0.0, 0

        for input_batch, target_batch in train_dl:
            input_batch = input_batch.float().to(device)
            target_batch = target_batch.float().to(device)

            optim.zero_grad()
            x_hat = decoder(input_batch)

            # Flexible loss function selection
            if Params.CNN_loss_function == "mse":
                loss = mse_loss(x_hat, target_batch)
            elif Params.CNN_loss_function == "mae":
                loss = mae_loss(x_hat, target_batch)
            elif Params.CNN_loss_function == "mse_mae":
                loss = mse_loss(x_hat, target_batch) + mae_loss(x_hat, target_batch)
            elif Params.CNN_loss_function == "smooth":
                loss_recon = mse_loss(x_hat, target_batch)
                loss_smooth = smoothness_loss(x_hat)
                loss = loss_recon + 0.01 * loss_smooth
            else:
                raise ValueError(f"Unknown loss function: {Params.CNN_loss_function}")

            loss.backward()
            if Params.CNN_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), Params.CNN_max_grad_norm)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Store train loss
        avg_train_loss = epoch_loss / max(1, n_batches)
        err_t.append(avg_train_loss)

        # --- Validation ---
        decoder.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for input_val, target_val in val_dl:
                input_val = input_val.float().to(device)
                target_val = target_val.float().to(device)
                x_hat_val = decoder(input_val)
                val_loss += mse_loss(x_hat_val, target_val).item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        err_val.append(avg_val_loss)

        scheduler.step()

        # Progress logging
        if (epoch + 1) % 1000 == 0 and Params.output:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{Params.CNN_epochs} "
                f"| Train: {avg_train_loss:.2e} "
                f"| Val: {avg_val_loss:.2e} "
                f"| POD: {pod_loss_ref:.2e} "
                f"| LR: {lr:.2e}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = deepcopy(decoder.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= Params.CNN_patience:
                if Params.output:
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(best val {best_val_loss:.3e})"
                    )
                break

    # Load best weights
    decoder.load_state_dict(best_model)

    # --- Plotting ---
    if Params.output:
        plt.figure(figsize=(8, 4))
        plt.semilogy(err_t, label="Train loss")
        plt.semilogy(err_val, label="Val loss")
        plt.axhline(
            y=pod_loss_ref, color="r", linestyle="--", label="POD baseline"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log)")
        plt.legend()
        plt.grid(which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return decoder, err_t, err_val
