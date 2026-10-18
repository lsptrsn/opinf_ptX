"""
Operators for Reduced Order Models (ROM) with Stability Guarantees.
Description:
    Defines the structural components of the ROM, including models with
    Global and Local stability guarantees based on Lyapunov theory,
    and a General model without constraints.

A note on the initial weights, because they turned out to matter more than a
starting point usually does.

The parameterisation A = J - R fixes the sign of the real parts of the
eigenvalues and says nothing about their magnitude. The loss, however,
advances the model one sampling interval with explicit Runge-Kutta, and that
step is stable only while |lambda| dt stays below 2*sqrt(2) = 2.83. Drawing
the entries at randn/10 puts the thermal case at |lambda| dt = 4.208 before
the first update, with two of four modes amplified by a factor of 8.16 per
step rather than damped. One step survives that. A fifty-step rollout does
not, which is why multi-step training diverges within a few epochs from this
starting point.

The same draw is also about three orders of magnitude above where the
operators have to end up: at dt = 10 s against hour-long dynamics the entries
settle near 1e-4, while the trained spectrum reaches |lambda| = 0.089 against
0.42 at the start. Adam has to cross that distance in a thousand steps, and
where it lands depends on where it began, which is one plausible route by
which the seed reaches the result.

Both are addressed by the two constants below. The defaults reproduce the
previous behaviour exactly.

A second note, on the size of the operators after training. The explicit RK4
multiplier R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24, z = lambda*dt, equals one
not only at z = 0 but at the three roots of z^3 + 4 z^2 + 12 z + 24 = 0,
namely z = -2.785 and z = -0.607 +/- 2.872i. Trained from |z| near two, the
operator places modes exactly there: to the one-step loss they look like
slow, almost constant states, while their exact solution decays by a factor
0.062 or 0.545 per sampling interval. The evaluation integrator follows the
exact solution and loses whatever dynamics the model stored in them.
A_MAX_RATE keeps the linear operator away from these roots. Once it does, the
quadratic term becomes the next route to the same fast dynamics: its Jacobian
H (q kron I + I kron q) acts as a state-dependent linear operator that the
bound on A does not see. H_MAX_RATE closes that route in the same units.
"""

__all__ = [
    "create_rom",
    "set_state_scale",
    "max_linearised_rate",
]

import numpy as np
import random
import torch
import torch.nn as nn
import opinf

# --- Global Configuration ---
device = opinf.utils.get_device()
Params = opinf.parameters.Params()


# --- initial weights ---------------------------------------------------------
# Scale of the initial operator entries. 0.1 is the randn/10 this replaces and
# is bit-identical to it; see _init_weight.
WEIGHT_INIT_SCALE = 0.05

# Target for the spectral radius of A at initialisation, in 1/s, or None to
# leave it wherever the draw puts it. None is the previous behaviour.
#
# When set, the factors of A are rescaled after drawing so that the largest
# |lambda| equals this value exactly. Two reasons to want that. It places the
# starting point inside the stability region of the training step by
# construction rather than by luck, which no choice of WEIGHT_INIT_SCALE can
# guarantee for every seed. And it removes the seed's influence on the size of
# the initial spectrum, leaving only its direction, which is the narrower
# question.
#
# A value to aim at: the trained model of the thermal case reaches
# |lambda| = 0.089 with dt = 10 s, so 0.05 starts just below where it ends and
# never crosses the limit on the way.
WEIGHT_INIT_SPECTRAL_RADIUS = None

# --- rate bound on the linear operator ----------------------------------------
# Upper bound on the size of A, in 1/s, or None for no bound. None is the
# previous behaviour and is bit-identical to it: the bound then performs no
# operation at all.
#
# Why a bound is needed. The one-step loss cannot distinguish a slow physical
# mode from a mode at one of the spurious roots of R(z) = 1 listed in the
# module docstring, because both have a one-step multiplier close to one. The
# spurious modes are reached from any start with |lambda|*dt near two, and a
# penalty on A does not remove them: between z = 0 and a spurious root the
# multiplier departs far from one (halfway to the complex root, |R| = 0.73 at
# a rotation of 78 degrees per step), so the loss forms a barrier the
# optimiser does not cross, whatever the penalty adds on the far side.
#
# When set, A is multiplied by a positive factor whenever its Frobenius norm
# exceeds the bound. A positive factor preserves the structure J - R R^T (and
# (J - R R^T) Q in the local model), so the stability guarantee is unchanged.
# The Frobenius norm is an upper bound on the spectral radius, so every
# eigenvalue satisfies |lambda| <= A_MAX_RATE. At 0.1 1/s and dt = 10 s that is
# |z| <= 1: a factor 2.8 below the smallest spurious root, and a range in which
# RK4 tracks exp(z) to about two percent. The Frobenius norm is used rather
# than the spectral norm because it needs no SVD, which keeps it cheap and
# available on every device.
#
# Applies to the global and the local model. The general model holds A as a
# raw parameter and is not bounded; create_rom says so when the bound is set.
# Not meant to be combined with WEIGHT_INIT_SPECTRAL_RADIUS, which would then
# rescale the already bounded operator.
A_MAX_RATE = 0.05


# Norm the rate bound on A acts on: "frobenius" (the previous behaviour,
# bit-identical) or "spectral". Both give |lambda| <= A_MAX_RATE for every
# eigenvalue. The Frobenius norm sums over all entries, so all modes share
# one budget and each added mode leaves less for the others; the spectral
# norm bounds the largest gain alone and leaves the other modes free,
# whatever the rank. The factor stays a positive scalar, so the structure
# J - R R^T and the stability guarantee are unchanged either way.
A_BOUND_NORM = "frobenius"

# Power iterations per call for the spectral norm. The singular vector is
# kept between calls, so a few iterations follow A as the optimiser moves
# it; the first call for a new shape runs _SPECTRAL_ITERS_COLD.
_SPECTRAL_ITERS = 5
_SPECTRAL_ITERS_COLD = 50
_spectral_vectors = {}

# --- rate bound on the quadratic operator -------------------------------------
# Upper bound on the rate the quadratic term can add to the linearised model,
# in 1/s, or None for no bound. None is the previous behaviour and is
# bit-identical to it.
#
# At a state q the linearisation of H (q kron q) is H (q kron I + I kron q),
# whose norm is at most 2 ||H||_F ||q||. With ||q|| bounded by the largest
# state norm in the training data, s = max_k ||q_k||, bounding
# ||H||_F <= H_MAX_RATE / (2 s) bounds the added rate by H_MAX_RATE at every
# training state. The linearised model then has |lambda| <= A_MAX_RATE +
# H_MAX_RATE there. Both are worst-case bounds; max_linearised_rate reports
# what the trained model actually reaches.
#
# A budget to aim at: A_MAX_RATE + H_MAX_RATE = 0.15 1/s gives |z| <= 1.5 at
# dt = 10 s, still well below the smallest spurious root at 2.785. RK4 loses
# accuracy on the negative real axis beyond |z| of about one (relative error
# 1.9 % at z = -1, 5.7 % at -1.2, 23 % at -1.5), but only for modes that
# decay within a few sampling intervals under either integrator.
#
# A positive factor preserves the energy-preserving structure of H in the
# global model, q^T H (q kron q) = 0. Applies to the global model only, which
# is the one that carries that structure. The bound needs the state scale and
# stays inactive until set_state_scale has been called on the model.
H_MAX_RATE = 0.05

# Added under the square root so that the norm has a finite gradient at zero.
# Far below any operator size that occurs; it changes no bounded value.
_NORM_EPS = 1e-24


def _init_weight(*shape):
    """Draw initial operator entries at WEIGHT_INIT_SCALE.

    The default keeps the division rather than multiplying by 0.1, because
    0.1 is not exactly representable and x / 10 differs from x * 0.1 in the
    last bit for some x. Bit-identity of the default matters more here than
    uniformity of the expression.
    """
    w = torch.randn(*shape)
    if WEIGHT_INIT_SCALE == 0.1:
        return w / 10
    return w * WEIGHT_INIT_SCALE


def _bound_rate(A):
    """Scale A down so that its Frobenius norm does not exceed A_MAX_RATE.

    Returns A itself, untouched, when no bound is set, so the default path is
    bit-identical to the unbounded model. Inside the bound the factor is
    exactly one; outside it the factor is A_MAX_RATE / ||A||_F, which keeps
    the direction of A and fixes its size.
    """
    if A_MAX_RATE is None:
        return A
    if A_BOUND_NORM == "spectral":
        norm = _spectral_norm(A)
    else:
        norm = torch.sqrt(torch.sum(A * A) + _NORM_EPS)
    limit = torch.as_tensor(float(A_MAX_RATE), dtype=A.dtype, device=A.device)
    return A * (limit / torch.maximum(norm, limit))


def _spectral_norm(A):
    """Largest singular value of A, by power iteration on A^T A.

    The singular vector is found without gradient and kept between calls,
    keyed by size, device and dtype, so each call needs only a few matrix
    products. The norm is then ||A v|| with that vector; once the iteration
    has converged, its gradient u v^T is the exact gradient of the largest
    singular value. Only matrix products are used, so it runs on MPS, where
    an SVD is not available. An unconverged vector underestimates the norm
    and loosens the bound by that amount, below 2 % in tests.
    """
    n = A.shape[1]
    key = (n, str(A.device), A.dtype)
    with torch.no_grad():
        v = _spectral_vectors.get(key)
        iters = _SPECTRAL_ITERS
        if v is None:
            v = torch.linspace(1.0, 2.0, n, dtype=A.dtype, device=A.device)
            v = v / torch.linalg.vector_norm(v)
            iters = _SPECTRAL_ITERS_COLD
        M = A.T @ A
        for _ in range(iters):
            v = M @ v
            v = v / (torch.linalg.vector_norm(v) + _NORM_EPS)
        _spectral_vectors[key] = v
    return torch.sqrt(torch.sum((A @ v) ** 2) + _NORM_EPS)


def _bound_quadratic(H, state_scale):
    """Scale H down so that the rate it adds stays below H_MAX_RATE.

    The limit on ||H||_F is H_MAX_RATE / (2 * state_scale); see H_MAX_RATE.
    Returns H itself, untouched, when no bound is set or when the state scale
    is not yet known, so the default path is bit-identical.
    """
    if H_MAX_RATE is None or state_scale is None:
        return H
    norm = torch.sqrt(torch.sum(H * H) + _NORM_EPS)
    limit = torch.as_tensor(float(H_MAX_RATE) / (2.0 * float(state_scale)),
                            dtype=H.dtype, device=H.device)
    return H * (limit / torch.maximum(norm, limit))


def set_state_scale(rom, states):
    """Record the largest state norm of the training data on the model.

    The rate bound on H is stated in 1/s and needs the size of the states the
    quadratic term acts on. Call once after create_rom with the training
    states, shape (r, k). Without this call H_MAX_RATE has no effect.

    Returns the recorded scale, or None when it could not be determined.
    """
    m = rom.module if hasattr(rom, "module") else rom
    q = np.asarray(states, dtype=float)
    scale = (float(np.max(np.linalg.norm(q, axis=0)))
             if q.ndim == 2 and q.size else float("nan"))
    m.state_scale = scale if np.isfinite(scale) and scale > 0 else None

    if Params.output and H_MAX_RATE is not None:
        if m.state_scale is None:
            print("  WARNING: H_MAX_RATE is set, but no usable state scale "
                  "was found; the quadratic term is not bounded.")
        elif getattr(m, "_H_tensor", None) is None:
            print("  rate bound on H: no energy-preserving H in this model, "
                  "nothing to bound")
        else:
            print(f"  rate bound on H: 2 ||H||_F max||q|| <= "
                  f"{H_MAX_RATE:.3e} 1/s, max||q|| = {m.state_scale:.4e}, "
                  f"so ||H||_F <= "
                  f"{H_MAX_RATE / (2.0 * m.state_scale):.4e}")
    return m.state_scale


def max_linearised_rate(rom, states):
    """Largest |lambda| of the linearised model over the given states, in 1/s.

    The linearisation of dq/dt = A q + H (q kron q) + ... at q is
    A + H (q kron I + I kron q). Without H this is the spectral radius of A,
    which the spectrum report already shows; with H it includes the rate the
    quadratic term adds, which that report does not see. States have shape
    (r, k), the layout of the training data.
    """
    m = rom.module if hasattr(rom, "module") else rom
    A = getattr(m, "A", None)
    if A is None:
        return float("nan")
    with torch.no_grad():
        A_np = A.detach().cpu().numpy().astype(float)
        H = getattr(m, "H", None)
        H_np = None if H is None else H.detach().cpu().numpy().astype(float)

    if H_np is None:
        ev = np.linalg.eigvals(A_np)
        return float(np.max(np.abs(ev))) if ev.size else float("nan")

    r = A_np.shape[0]
    q = np.asarray(states, dtype=float)
    if q.ndim != 2 or q.shape[0] != r:
        raise ValueError(f"states must have shape ({r}, k), got {q.shape}")

    # H3[i, b, d] multiplies q_b q_d, matching the ordering of _kron. The
    # derivative of sum_{b,d} H3[i,b,d] q_b q_d with respect to q_j is
    # sum_d H3[i,j,d] q_d + sum_b H3[i,b,j] q_b, evaluated at every state.
    H3 = H_np.reshape(r, r, r)
    J = (A_np[None, :, :]
         + np.einsum("ijd,dn->nij", H3, q)
         + np.einsum("ibj,bn->nij", H3, q))
    ev = np.linalg.eigvals(J)
    return float(np.max(np.abs(ev)))


def _spectral_radius(rom):
    """Largest |lambda| of the linear operator, or NaN when there is none."""
    A = rom.A
    if A is None:
        return float("nan")
    with torch.no_grad():
        try:
            ev = np.linalg.eigvals(A.detach().cpu().numpy())
        except np.linalg.LinAlgError:
            return float("nan")
    return float(np.max(np.abs(ev))) if ev.size else float("nan")


def _rescale_spectrum(rom, target):
    """Scale the factors of A so that its spectral radius equals ``target``.

    A = J - R with J linear in _J and R quadratic in _R, so scaling _J by c
    and _R by sqrt(c) scales both parts by c and the spectrum with them. The
    local variant carries a separate factor Q, which is left alone, so the
    same argument applies there. A general model holds A directly.

    When A_MAX_RATE is set, the radius seen here is that of the bounded
    operator.

    Returns the radius before rescaling, for reporting.
    """
    before = _spectral_radius(rom)
    if not np.isfinite(before) or before <= 0 or target <= 0:
        return before

    c = float(target) / before
    with torch.no_grad():
        if getattr(rom, "_J", None) is not None:
            rom._J.mul_(c)
            rom._R.mul_(float(np.sqrt(c)))
        elif isinstance(getattr(rom, "A", None), torch.nn.Parameter):
            rom.A.mul_(c)
        else:
            return before
    return before


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
            sys_order=sys_order, has_A=has_A, has_B=has_B,
            has_C=has_C, has_H=has_H
        )
    elif Params.stability == 'local':
        rom = _ModelHypothesisLocalStable(
            sys_order=sys_order, has_A=has_A, has_B=has_B,
            has_C=has_C, has_H=has_H
        )
    else:
        rom = _GeneralModel(
            sys_order=sys_order, has_A=has_A, has_B=has_B,
            has_C=has_C, has_H=has_H
        )

    # No state scale yet: the bound on H stays inactive until
    # set_state_scale is called with the training states.
    rom.state_scale = None

    # State the bounds in the log, so that every run documents whether they
    # were active. A log that does not say so cannot be compared with one
    # that does.
    if A_MAX_RATE is not None and Params.output:
        if Params.stability in ('global', 'local'):
            print(f"  rate bound on A: {A_BOUND_NORM} norm <= {A_MAX_RATE:.3e} "
                  f"1/s, so every |lambda| <= {A_MAX_RATE:.3e} 1/s")
        else:
            print("  WARNING: A_MAX_RATE is set, but the general model holds "
                  "A as a raw parameter; no bound is applied.")
    if H_MAX_RATE is not None and Params.output \
            and Params.stability != 'global' and has_H:
        print("  WARNING: H_MAX_RATE is set, but only the global model is "
              "bounded; no bound is applied to H.")

    if WEIGHT_INIT_SPECTRAL_RADIUS is not None:
        before = _rescale_spectrum(rom, WEIGHT_INIT_SPECTRAL_RADIUS)
        if Params.output and np.isfinite(before):
            print(f"  initial spectrum rescaled: max |lambda| "
                  f"{before:.4e} -> {WEIGHT_INIT_SPECTRAL_RADIUS:.4e} 1/s")

    rom = rom.to(device=device, dtype=torch.float32)

    # Wrap for multi-GPU support if available.
    # Note: Remove DataParallel if strict determinism issues occur.
    if torch.cuda.device_count() > 1:
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
            self._J = torch.nn.Parameter(_init_weight(sys_order, sys_order))
            self._R = torch.nn.Parameter(_init_weight(sys_order, sys_order))
            operators.append('Aq(t)')
        else:
            self._J = self._R = None

        if has_B:
            self.B = torch.nn.Parameter(
                _init_weight(sys_order, Params.input_dim))
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(_init_weight(sys_order, 1))
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
        """Constructs stable A = (J - R), bounded in size when A_MAX_RATE is set."""
        if self._J is None or self._R is None:
            return None
        J = self._J - self._J.T
        R = self._R @ self._R.T
        return _bound_rate(J - R)

    @property
    def H(self):
        """Constructs the energy-preserving quadratic operator.

        Bounded in the rate it can add when H_MAX_RATE is set and the state
        scale is known.
        """
        if self._H_tensor is None:
            return None
        H_perm = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - H_perm
        H = J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)
        return _bound_quadratic(H, getattr(self, "state_scale", None))

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
            self._J = torch.nn.Parameter(_init_weight(sys_order, sys_order))
            self._R = torch.nn.Parameter(_init_weight(sys_order, sys_order))
            self._Q = torch.nn.Parameter(_init_weight(sys_order, sys_order))
            operators.append('Aq(t)')
        else:
            self._J = self._R = self._Q = None

        if has_B:
            self.B = torch.nn.Parameter(
                _init_weight(sys_order, Params.input_dim))
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(_init_weight(sys_order, 1))
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
        """Constructs locally stable A, bounded in size when A_MAX_RATE is set."""
        if self._J is None or self._R is None:
            return None
        J = self._J - self._J.T
        R = self._R @ self._R.T
        Q = self._Q @ self._Q.T

        if Params.local_set_Q_to_identity:
            return _bound_rate(J - R)  # Q assumed to be Identity to simplify learning
        return _bound_rate((J - R) @ Q)

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


class _GeneralModel(nn.Module):
    """
    Standard ROM implementation without stability constraints.
    Learns raw A, B, C, and H operators directly.
    """

    def __init__(self, sys_order, has_A, has_B, has_C, has_H):
        super().__init__()
        if Params.output:
            print("Model Initialization: no stability constraint.")
        operators = []

        if has_A:
            self.A = torch.nn.Parameter(_init_weight(sys_order, sys_order))
            operators.append('Aq(t)')
        else:
            self.A = None

        if has_B:
            self.B = torch.nn.Parameter(
                _init_weight(sys_order, Params.input_dim))
            operators.append('Bu')
        else:
            self.B = None

        if has_C:
            self.C = torch.nn.Parameter(_init_weight(sys_order, 1))
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