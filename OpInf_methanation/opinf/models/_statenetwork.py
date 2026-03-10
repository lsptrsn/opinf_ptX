"""
Neural Network Architectures for OpInf PINNs
Description:
    Implementation of various neural network architectures, including
    Standard MLPs and SIREN (Sinusoidal Representation Networks),
    optimized for Physics-Informed Neural Networks.
    Inspiration drawn from the DeePyMoD package: https://github.com/PhIMaL/DeePyMoD
"""

__all__ = [
    "FeedForwardNN",
    "Siren",
    "create_network"
]

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
import random
import opinf

# --- Global Configuration ---
device = opinf.utils.get_device()
Params = opinf.parameters.Params()

# Set seeds for reproducibility
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# ==============================================================
#  Standard Feed-Forward NN (MLP)
# ==============================================================

class LambdaLayer(nn.Module):
    """Simple helper layer to apply an arbitrary function in a Sequential model."""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Sin(nn.Module):
    """Sinusoidal activation function."""
    def forward(self, x):
        return torch.sin(x)


class FeedForwardNN(nn.Module):
    """
    Feed-forward neural network (MLP) with configurable hidden layers and activation.
    Used as the backbone for standard PINN approaches.
    """
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int):
        super().__init__()

        act_name = Params.PINN_activation.lower()

        # Select activation function based on configuration
        if act_name == 'selu':
            activation = nn.SELU()
        elif act_name == 'tanh':
            activation = nn.Tanh()
        elif act_name in ('sin', 'sine'):
            activation = Sin()
        else:
            raise ValueError(f"Activation function '{Params.PINN_activation}' not supported.")

        # Build layer structure
        layers = []
        architecture = [n_in] + n_hidden + [n_out]

        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            # Add activation to all but the last layer
            if i < len(architecture) - 2:
                layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both the output and the input for gradient computation in PINNs."""
        y = self.network(x)
        return y, x


# ==============================================================
#  SIREN Network (Sinusoidal Representation Networks)
# ==============================================================

class SineLayer(nn.Module):
    """
    Linear layer followed by a sine activation, with specific weight
    initialization for preserving distribution across layers.
    Ref: Sitzmann et al., 2020.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """
    SIREN model implementation for implicit neural representations.
    Particularly effective for high-frequency signals and derivatives.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: List[int],
        n_out: int,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.network = self.build_network(
            n_in, n_hidden, n_out, first_omega_0, hidden_omega_0
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the network output and input for derivative calculations."""
        return self.network(input), input

    def build_network(
        self,
        n_in: int,
        n_hidden: List[int],
        n_out: int,
        first_omega_0: float,
        hidden_omega_0: float,
    ) -> torch.nn.Sequential:
        network = []

        # Input layer
        network.append(SineLayer(n_in, n_hidden[0], is_first=True, omega_0=first_omega_0))

        # Hidden layers
        for layer_i, layer_j in zip(n_hidden, n_hidden[1:]):
            network.append(SineLayer(layer_i, layer_j, omega_0=hidden_omega_0))

        # Output layer
        final_linear = nn.Linear(n_hidden[-1], n_out)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / n_hidden[-1]) / hidden_omega_0,
                np.sqrt(6 / n_hidden[-1]) / hidden_omega_0,
            )
        network.append(final_linear)

        return nn.Sequential(*network)

# ==============================================================
#  Factory Function
# ==============================================================

def create_network(n_in: int, n_hidden: List[int], n_out: int) -> nn.Module:
    """
    Factory function to instantiate the neural network architecture
    specified in Params.PINN_architecture.

    Supported Architectures:
        - 'feedforward': Standard Multi-Layer Perceptron.
        - 'siren': Sinusoidal Representation Network.
    """
    arch = Params.PINN_architecture.lower()

    if arch == "feedforward":
        return FeedForwardNN(n_in, n_hidden, n_out)
    elif arch == "siren":
        return Siren(n_in, n_hidden, n_out)
    else:
        raise ValueError(f"Unknown architecture '{Params.PINN_architecture}'. "
                         "Choose from ['feedforward', 'siren'].")
