"""
Operators for Reduced Order Models (ROM) with Stability Guarantees.
Description:
    Defines the structural components of the ROM, including models with
    Global and Local stability guarantees based on Lyapunov theory,
    and a General model without constraints.
"""

__all__ = [
    "create_rom"
]

import numpy as np
import random
import torch
import torch.nn as nn
import opinf

# --- Global Configuration ---
device = opinf.utils.get_device()
Params = opinf.parameters.Params()


def set_seed(seed: int = 0):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    Crucial for consistent weight initialization across different trials
    (e.g., Optuna optimization).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Force deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_rom(non_markov: bool = False, seed: int = 0) -> nn.Module:
    """
    Factory function to initialize a ROM based on the specified stability
    type and model structure.

    Parameters:
        non_markov (bool): If True, enforces a specific structure for
                           Non-Markovian ROMs (A and B only).
        seed (int): Random seed for reproducible weight initialization.

    Returns:
        torch.nn.Module: The initialized and device-mapped ROM.
    """
    # Set seed here to ensure identical initialization for every call
    # (important for hyperparameter optimization trials).
    set_seed(seed)

    # Identify components from the model structure string (e.g., 'ABH')
    has_A = 'A' in Params.model_structure
    has_B = 'B' in Params.model_structure
    has_C = 'C' in Params.model_structure
    has_H = 'H' in Params.model_structure

    if non_markov:
        has_A, has_B = True, True
        has_C, has_H = False, False

    sys_order = Params.ROM_order

    # Instantiate the requested model type
    if Params.stability == 'global':
        rom = _ModelHypothesisGlobalStable(
            sys_order=sys_order, has_A=has_A, has_B=has_B, has_C=has_C, has_H=has_H
        ).double()
    elif Params.stability == 'local':
        rom = _ModelHypothesisLocalStable(
            sys_order=sys_order, has_A=has_A, has_B=has_B, has_C=has_C, has_H=has_H
        ).double()
    else:
        rom = _GeneralModel(
            sys_order=sys_order, has_A=has_A, has_B=has_B, has_C=has_C, has_H=has_H
        ).double()

    # Move to target device
    rom = rom.to(device=device, dtype=torch.float32)

    # Wrap for multi-GPU support if available.
    # Note: Remove DataParallel if strict determinism issues occur.
    rom = torch.nn.DataParallel(rom)

    return rom


class _ModelHypothesisGlobalStable(nn.Module):
    """
    ROM with Global Stability Guarantees.
    The linear operator A is parameterized as A = (J - R), where J is
    skew-symmetric and R is positive semi-definite, ensuring that
    the eigenvalues of A have non-positive real parts.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H):
        super().__init__()
        self.sys_order = sys_order
        operators = []

        if Params.output:
            print("Model Initialization: Global Stability Guarantees enabled.")

        if has_A:
            # Parameters used to construct the stable A matrix
            self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            operators.append('Aq(t)')
        else:
            self._J = self._R = None

        if has_B:
            self.B = torch.nn.Parameter(torch.randn(sys_order, Params.input_dim) / 10)
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(torch.randn(sys_order, 1) / 10)
            operators.append('C')
        else:
            self.C = None

        if has_H:
            self._H_tensor = torch.nn.Parameter(torch.zeros(sys_order, sys_order, sys_order))
            operators.append('H[q(t) ⊗ q(t)]')
        else:
            if Params.output:
                print('Warning: Global stability usually requires a quadratic term (H).')
            self._H_tensor = None

        if Params.output:
            print(f"ROM structure: dq/dt = {' + '.join(operators)}")

    @property
    def A(self):
        """Constructs stable A = (J - R) from learned parameters."""
        if self._J is None or self._R is None:
            return None
        J = self._J - self._J.T
        R = self._R @ self._R.T
        return J - R

    @property
    def H(self):
        """Constructs the quadratic operator from the learned tensor."""
        if self._H_tensor is None:
            return None
        H_perm = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - H_perm
        return J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)

    def forward(self, x, t, u):
        model = torch.zeros_like(x)
        if self.A is not None:
            model += x @ self.A.T
        if self.H is not None:
            model += _kron(x, x) @ self.H.T
        if self.C is not None:
            model += self.C.T
        if self.B is not None and u is not None:
            model += u @ self.B.T
        return model


class _ModelHypothesisLocalStable(nn.Module):
    """
    ROM with Local Stability Guarantees.
    Includes an additional learned Q matrix for the parameterization
    A = (J - R)Q.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H):
        super().__init__()
        operators = []

        if Params.output:
            print("Model Initialization: Local Stability Guarantees enabled.")

        if has_A:
            self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            self._Q = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            operators.append('Aq(t)')

        if has_B:
            self.B = torch.nn.Parameter(torch.randn(sys_order, Params.input_dim) / 10)
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(torch.randn(sys_order, 1) / 10)
            operators.append('C')
        else:
            self.C = None

        if has_H:
            self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2))
            operators.append('H[q(t) ⊗ q(t)]')
        else:
            self.H = None

        if Params.output:
            print(f"ROM structure: dq/dt = {' + '.join(operators)}")

    @property
    def A(self):
        """Constructs locally stable A matrix."""
        if self._J is None or self._R is None:
            return None
        J = self._J - self._J.T
        R = self._R @ self._R.T
        Q = self._Q @ self._Q.T

        if Params.local_set_Q_to_identity:
            return J - R  # Q assumed to be Identity to simplify learning
        return (J - R) @ Q

    def forward(self, x, t, u):
        model = torch.zeros_like(x)
        if self.A is not None:
            model += x @ self.A.T
        if self.H is not None:
            model += _kron(x, x) @ self.H.T
        if self.C is not None:
            model += self.C.T
        if self.B is not None:
            model += u @ self.B.T
        return model


class _GeneralModel(nn.Module):
    """
    Standard ROM implementation without stability constraints.
    Learns raw A, B, C, and H operators directly.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H):
        super().__init__()
        operators = []

        if has_A:
            self.A = torch.nn.Parameter(torch.randn(sys_order, sys_order) / 10)
            operators.append('Aq(t)')
        else:
            self.A = None

        if has_B:
            self.B = torch.nn.Parameter(torch.randn(sys_order, Params.input_dim) / 10)
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(torch.randn(sys_order, 1) / 10)
            operators.append('C')
        else:
            self.C = None

        if has_H:
            self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2))
            operators.append('H[q(t) ⊗ q(t)]')
        else:
            self.H = None

        if Params.output:
            print(f"ROM structure: dq/dt = {' + '.join(operators)}")
            print("Stability: None (General Model)")

    def forward(self, x, t, u):
        model = torch.zeros_like(x)
        if self.A is not None:
            model += x @ self.A.T
        if self.H is not None:
            model += _kron(x, x) @ self.H.T
        if self.C is not None:
            model += self.C.T
        if self.B is not None and u is not None:
            model += u @ self.B.T
        return model


def _kron(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Efficiently computes the Kronecker product for a batch of vectors.
    Used for the quadratic term H[q ⊗ q].
    """
    return torch.einsum("ab,ad->abd", [x, y]).view(x.size(0), x.size(1) * y.size(1))
