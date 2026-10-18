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
import opinf

# --- Global Configuration ---
device = opinf.utils.get_device()

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
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int, activation_type: str = 'gelu'):
        super().__init__()

        act_name = activation_type.lower()

        # Select activation function based on configuration
        if act_name == 'selu':
            activation = nn.SELU()
        elif act_name == 'tanh':
            activation = nn.Tanh()
        elif act_name in ('sin', 'sine'):
            activation = Sin()
        elif act_name == 'gelu':
            activation = nn.GELU()
        elif act_name == 'silu':
            activation = nn.SiLU()
        elif act_name == 'softplus':
            activation = nn.Softplus()
        else:
            raise ValueError(f"Activation function '{activation_type}' not supported.")

        # Build layer structure
        layers = []
        architecture = [n_in] + list(n_hidden) + [n_out]

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
    Factory function to instantiate the neural network architecture.
    Handles conditional activation logic cleanly.
    """
    Params = opinf.parameters.Params()
    arch = Params.PINN_architecture.lower()

    if getattr(Params, 'output', False):
        msg = f"INFO: Creating [{arch}] architecture"
        if arch == "feedforward":
            msg += f" with activation: {Params.PINN_activation}"
        print(msg)

    # 2. Netzwerk erstellen
    if arch == "feedforward":
        net = FeedForwardNN(n_in, n_hidden, n_out, activation_type=Params.PINN_activation)
    elif arch == "siren":
        net = Siren(n_in, n_hidden, n_out)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from ['feedforward', 'siren'].")
    if arch == "siren" and getattr(Params, 'output', False):
        print(f"INFO: Creating [siren] architecture; "
              f"PINN_activation = '{Params.PINN_activation}' is not used "
              f"by SIREN, which has sine activations by construction.")

    return net
