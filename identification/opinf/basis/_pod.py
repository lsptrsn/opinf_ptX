"""
Tools for Basis Computation and Reduced-Dimension Selection.
This module provides functions for Proper Orthogonal Decomposition (POD),
nonlinear manifold learning via polynomial expansions, and alternating
minimization for dimensionality reduction.
"""

__all__ = [
    "pod",
    "polynomial_form",
    "get_basis_and_reduced_data",
    "basis_multi",
    "basis_nonlin_multi",
    "svdval_decay",
    "cumulative_energy",
    "svd_results",
    "residual_energy",
    "MANIFOLD_DEGREE",
]

import os
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sklearn.utils.extmath as sklmath

import opinf.parameters
# Deliberately not ``import opinf.post``. Nothing here uses it, and the package
# imports basis before post, so that line made basis/_pod.py pull a
# half-initialised post back into itself. Any module that needs something from
# post has to reach for it inside a function, not at import time.

# Initialize global parameters from dataclass
Params = opinf.parameters.Params()

# Where the diagnostic figures of this module are written. Every writer below
# creates it; two of them used to assume that cumulative_energy had run first
# and raised FileNotFoundError when it had not.
FIGURE_DIR = './results/figures'

###############################################################################
# VISUALIZATION CONFIGURATION (Institute Color Palette)
###############################################################################
mpi_colors = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255)
}

###############################################################################
# MANIFOLD HYPERPARAMETERS
###############################################################################
MANIFOLD_DEGREE = 3       # p in the polynomial embedding g(q) = [q^2, ..., q^p]
MANIFOLD_GAMMA = 1e-6     # Tikhonov parameter of the Xi regression
MANIFOLD_Q_MAX = 100      # hard cap on correction modes per field
MANIFOLD_Q_TOL = 1e-3     # relative singular value taken as the noise floor


def _select_q(block, r, n_available, name):
    """Number of correction modes for one physical field.

    The correction modes are the singular vectors ranked r+1 ... r+q. Admitting
    every remaining mode means fitting the reconstruction map to whatever sits
    in the noise floor of the record. Geelen et al. document that failure mode
    explicitly: the trailing singular vectors carry near-zero singular values,
    the extra terms buy nothing, and prediction outside the training window
    degrades. On experimental data the trailing modes are sensor noise, so q is
    chosen from the spectrum instead of fixed.

    A field with no retained linear modes contributes nothing to the latent
    state and therefore gets no correction modes either.
    """
    if r <= 0 or block.size == 0 or n_available <= 0:
        if Params.output:
            print(f"  q[{name}] = 0 (field carries no linear modes)")
        return 0

    svdvals = np.linalg.svd(block, compute_uv=False)
    if svdvals.size == 0 or svdvals[0] <= 0:
        if Params.output:
            print(f"  q[{name}] = 0 (block is numerically zero)")
        return 0

    n_above = int(np.count_nonzero(svdvals / svdvals[0] >= MANIFOLD_Q_TOL))
    q = int(min(MANIFOLD_Q_MAX, n_available, max(0, n_above - r)))

    if Params.output:
        print(f"  q[{name}] = {q}  ({n_above} modes above "
              f"{MANIFOLD_Q_TOL:.0e} of sigma_1, r = {r}, "
              f"{n_available} available)")
    return q

def pod(states, r: int = "full", mode: str = "dense", return_W: bool = False, **options):
    """
    Compute the Proper Orthogonal Decomposition (POD) basis of rank r.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots. Each column is a snapshot of dimension n.
    r : int or "full"
        Number of POD basis vectors to compute. Defaults to full SVD rank.
    mode : str
        SVD computation strategy:
        * "dense": Standard SVD via scipy.linalg.svd.
        * "randomized": Approximate SVD via sklearn.utils.extmath.randomized_svd.
    return_W : bool
        If True, returns the right singular vectors (coefficients).
    **options
        Keyword arguments passed to the respective SVD solver.

    Returns
    -------
    V : (n, r) ndarray
        Left singular vectors (POD basis).
    svdvals : (r,) ndarray
        Singular values in descending order.
    W : (k, r) ndarray, optional
        Right singular vectors. Only returned if return_W is True.
    """
    rmax = min(states.shape)
    if r == "full":
        r = rmax
    if r > rmax or r < 1:
        raise ValueError(f"Invalid POD rank r = {r} (must satisfy 1 ≤ r ≤ {rmax})")

    if mode in ["dense", "simple"]:
        V, svdvals, Wt = la.svd(states, full_matrices=False, **options)
        W = Wt.T
    elif mode == "randomized":
        options.setdefault("random_state", None)
        V, svdvals, Wt = sklmath.randomized_svd(states, r, **options)
        W = Wt.T
    else:
        raise NotImplementedError(f"Mode '{mode}' not recognized.")

    if return_W:
        return V[:, :r], svdvals, W[:, :r]
    return V[:, :r], svdvals

def polynomial_form(x, p=MANIFOLD_DEGREE):
    """
    Generate polynomial expansion terms for a given input.

    Parameters
    ----------
    x : ndarray
        Input data array (reduced state).
    p : int
        Maximum degree of the polynomial expansion.

    Returns
    -------
    list
        List of expanded terms [x^2, x^3, ..., x^p].
    """
    return [x**degree for degree in range(2, p + 1)]

def _rel_fro(exact, approx):
    """Relative Frobenius-norm error, NaN when the reference is zero."""
    denom = float(np.linalg.norm(exact, 'fro'))
    if not np.isfinite(denom) or denom <= 0:
        return float('nan')
    return float(np.linalg.norm(exact - approx, 'fro')) / denom


def _report_reconstruction(Q_shifted, states_repro, V_X, V_T, label):
    """Report the relative projection error, per field and over the whole.

    Both arguments are in the shifted and scaled snapshot space, so the
    denominator is the centred snapshot matrix. That is the relative state
    error of Geelen et al. Putting the reference state back into the
    denominator, as an earlier version did, measures the residual against the
    mean operating point instead and understates the error by three orders of
    magnitude.

    The number over the concatenated matrix is reported for continuity, but it
    is not a physical quantity: each field is divided by its own maximum before
    the basis is computed, so the weight a field carries in the combined norm
    is a preprocessing choice. The per-field errors do not have that problem,
    since the factor cancels between numerator and denominator, and their mean
    is the same combination that post/_errors.py forms for the model error.
    That mean is the number to hold against the model error; anything else
    compares two differently weighted averages.
    """
    combined = _rel_fro(Q_shifted, states_repro)

    n_X, n_T = V_X.shape[0], V_T.shape[0]
    err_F = _rel_fro(Q_shifted[:n_X, :], states_repro[:n_X, :])
    err_T = _rel_fro(Q_shifted[n_X:n_X + n_T, :],
                     states_repro[n_X:n_X + n_T, :])
    per_field = [e for e in (err_F, err_T) if np.isfinite(e)]
    mean = float(np.mean(per_field)) if per_field else float('nan')

    if Params.output:
        print(f"{label} reconstruction error: {combined:.4%} "
              f"over the concatenated matrix")
        print(f"  per field: F {err_F:.4%}, T {err_T:.4%}, "
              f"mean {mean:.4%}  (the mean is what the model error "
              f"is comparable with)")

    return combined

def _linear_pod_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states):
    """
    Perform multi-variable linear POD reduction.

    The method constructs a block-diagonal basis from individual variable
    SVDs and projects the shifted dataset.
    """
    # Retrieve target ranks for each variable
    r_F, r_T, r_w1, r_w2 = Params.r_F, Params.r_T, Params.r_w1, Params.r_w2

    # Construct global reduced basis
    V_reduced = basis_multi(V_X, r_F, V_T, r_T, V_w1, r_w1, V_w2, r_w2)

    # Dimensionality reduction and reconstruction
    Q_reduced = opinf.utils.reduced_state(Q_shifted, V_reduced)
    states_repro = V_reduced @ Q_reduced
    _report_reconstruction(Q_shifted, states_repro, V_X, V_T, "Linear POD")

    return Q_reduced, V_reduced, None, None

def _nonlinear_pod_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states):
    """
    Compute a Nonlinear POD basis using polynomial expansion.

    Following the methodology of Geelen et al. (2023), this implementation
    augments the linear subspace with a polynomial manifold to capture
    nonlinear correlations in experimental data.
    """
    # 1. Initialize with Linear POD
    Q_reduced, V_reduced, _, _ = _linear_pod_basis(
        V_X, V_T, V_w1, V_w2, Q_shifted, reference_states
    )

    # 2. Hyperparameters for manifold construction
    p, gamma = MANIFOLD_DEGREE, MANIFOLD_GAMMA

    # 3. Correction modes, chosen per field from the singular value spectrum
    r_F, r_T, r_w1, r_w2 = Params.r_F, Params.r_T, Params.r_w1, Params.r_w2
    n_X, n_T, n_w1 = V_X.shape[0], V_T.shape[0], V_w1.shape[0]

    if Params.output:
        print(f"NL-POD setup: p={p}, gamma={gamma:.1e}, "
              f"noise floor {MANIFOLD_Q_TOL:.0e}")

    q_X = _select_q(Q_shifted[:n_X, :], r_F, V_X.shape[1] - r_F, 'F')
    q_T = _select_q(Q_shifted[n_X:n_X + n_T, :], r_T,
                    V_T.shape[1] - r_T, 'T')
    q_w1 = _select_q(Q_shifted[n_X + n_T:n_X + n_T + n_w1, :], r_w1,
                     V_w1.shape[1] - r_w1, 'w1')
    q_w2 = _select_q(Q_shifted[n_X + n_T + n_w1:, :], r_w2,
                     V_w2.shape[1] - r_w2, 'w2')

    # 4. Construct V_bar (Nonlinear basis extension)
    V_reduced_nonlin = basis_nonlin_multi(
        V_X, r_F, q_X, V_T, r_T, q_T, V_w1, r_w1, q_w1, V_w2, r_w2, q_w2
    )

    # 5. Regression for Manifold Mapping (Xi)
    # Project linear residual onto the nonlinear basis extension
    proj_error = Q_shifted - (V_reduced @ Q_reduced)
    poly = np.concatenate(polynomial_form(Q_reduced, p), axis=0)

    # Solve the regularized least-squares problem for Xi:
    # Xi = (V_bar^T * Error * Poly^T) * inv(Poly * Poly^T + gamma * I)
    RHS = (V_reduced_nonlin.T @ proj_error) @ poly.T
    LHS = poly @ poly.T + gamma * np.identity(poly.shape[0])

    try:
        Xi = np.linalg.solve(LHS, RHS.T).T
    except np.linalg.LinAlgError:
        if Params.output:
            print("Warning: Matrix singular. Utilizing Moore-Penrose pseudo-inverse.")
        Xi = RHS @ np.linalg.pinv(LHS)

    # 6. Reconstruction and Error Analysis
    states_repro_NLPOD = V_reduced @ Q_reduced + V_reduced_nonlin @ Xi @ poly
    _report_reconstruction(Q_shifted, states_repro_NLPOD, V_X, V_T,
                           "Nonlinear POD")

    return Q_reduced, V_reduced, V_reduced_nonlin, Xi

def _nonlinear_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states):
    """
    Compute nonlinear manifold via Alternating Minimization (AM).

    Iteratively optimizes the basis matrices (via Orthogonal Procrustes)
    and the reduced states (via Nonlinear Least Squares) to minimize the
    reconstruction error.
    """
    # Configuration
    p, gamma = MANIFOLD_DEGREE, MANIFOLD_GAMMA
    max_iter = 20
    tol = 1e-4
    
    r_F, r_T, r_w1, r_w2 = Params.r_F, Params.r_T, Params.r_w1, Params.r_w2
    n_X, n_T, n_w1 = V_X.shape[0], V_T.shape[0], V_w1.shape[0]
    
    q_X = _select_q(Q_shifted[:n_X, :], r_F, V_X.shape[1] - r_F, 'F')
    q_T = _select_q(Q_shifted[n_X:n_X + n_T, :], r_T,
                    V_T.shape[1] - r_T, 'T')
    q_w1 = _select_q(Q_shifted[n_X + n_T:n_X + n_T + n_w1, :], r_w1,
                     V_w1.shape[1] - r_w1, 'w1')
    q_w2 = _select_q(Q_shifted[n_X + n_T + n_w1:, :], r_w2,
                     V_w2.shape[1] - r_w2, 'w2')

    # Initialization via Linear POD and NL-POD guess
    Q_reduced, V_reduced, _, _ = _linear_pod_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states)
    V_reduced_nonlin = basis_nonlin_multi(V_X, r_F, q_X, V_T, r_T, q_T, V_w1, r_w1, q_w1, V_w2, r_w2, q_w2)

    proj_error = Q_shifted - (V_reduced @ Q_reduced)
    poly = np.concatenate(polynomial_form(Q_reduced, p), axis=0)

    # Initial estimate of Xi
    M = poly @ poly.T + gamma * np.eye(poly.shape[0])
    RHS_init = V_reduced_nonlin.T @ proj_error @ poly.T
    Xi = np.linalg.solve(M, RHS_init.T).T if np.linalg.cond(M) < 1/np.finfo(float).eps else RHS_init @ np.linalg.pinv(M)

    # Domain-specific variable indices for block-diagonal updates
    id_rX, id_rT, id_rw1 = r_F, r_F + r_T, r_F + r_T + r_w1

    X_s = Q_shifted[:n_X, :]
    T_s = Q_shifted[n_X : n_X+n_T, :]
    w1_s = Q_shifted[n_X+n_T : n_X+n_T+n_w1, :]
    w2_s = Q_shifted[n_X+n_T+n_w1 :, :]

    def obj_func(q_vec, snapshot_idx):
        """Residual objective function for nonlinear state optimization."""
        q_reshaped = q_vec.reshape(-1, 1)
        poly_terms = np.concatenate(polynomial_form(q_reshaped, p), axis=0).flatten()
        recon = V_reduced @ q_vec + V_reduced_nonlin @ Xi @ poly_terms
        return Q_shifted[:, snapshot_idx] - recon

    if Params.output:
        print("*** Starting Alternating Minimization Procedure ***")

    nrg_old = 0.0
    num_snapshots = Q_shifted.shape[1]

    for niter in range(max_iter):
        # --- Step 1: Block-wise Basis Update (Orthogonal Procrustes) ---
        Xi_X, Xi_T = Xi[:q_X, :], Xi[q_X:q_X+q_T, :]
        Xi_w1, Xi_w2 = Xi[q_X+q_T:q_X+q_T+q_w1, :], Xi[q_X+q_T+q_w1:, :]

        # Update each variable block independently to preserve block-diagonal structure
        for block_id in ['X', 'T', 'w1', 'w2']:
            if block_id == 'X':
                Z, S_block = np.vstack([Q_reduced[:id_rX, :], Xi_X @ poly]), X_s
            elif block_id == 'T':
                Z, S_block = np.vstack([Q_reduced[id_rX:id_rT, :], Xi_T @ poly]), T_s
            elif block_id == 'w1':
                Z, S_block = np.vstack([Q_reduced[id_rT:id_rw1, :], Xi_w1 @ poly]), w1_s
            else:
                Z, S_block = np.vstack([Q_reduced[id_rw1:, :], Xi_w2 @ poly]), w2_s

            Um, _, Vm = np.linalg.svd(S_block @ Z.T, full_matrices=False)
            if block_id == 'X': Om_X = Um @ Vm
            elif block_id == 'T': Om_T = Um @ Vm
            elif block_id == 'w1': Om_w1 = Um @ Vm
            else: Om_w2 = Um @ Vm

        # Update global basis matrices
        V_reduced = basis_multi(Om_X, r_F, Om_T, r_T, Om_w1, r_w1, Om_w2, r_w2)
        V_reduced_nonlin = basis_nonlin_multi(Om_X, r_F, q_X, Om_T, r_T, q_T, Om_w1, r_w1, q_w1, Om_w2, r_w2, q_w2)

        # --- Step 2: Manifold Mapping Update (Xi) ---
        proj_error = Q_shifted - (V_reduced @ Q_reduced)
        M_Xi = poly @ poly.T + gamma * np.eye(poly.shape[0])
        RHS_Xi = V_reduced_nonlin.T @ proj_error @ poly.T
        Xi = np.linalg.solve(M_Xi, RHS_Xi.T).T

        # --- Step 3: Reduced State Optimization (Nonlinear Least Squares) ---
        for j in range(num_snapshots):
            res = opt.least_squares(obj_func, Q_reduced[:, j], args=(j,),
                                   ftol=1e-4, xtol=1e-4, max_nfev=20)
            Q_reduced[:, j] = res.x

        poly = np.concatenate(polynomial_form(Q_reduced, p), axis=0)

        # Convergence Monitoring
        recon_full = V_reduced @ Q_reduced + V_reduced_nonlin @ Xi @ poly
        energy = np.linalg.norm(recon_full, 'fro')**2 / np.linalg.norm(Q_shifted, 'fro')**2
        if abs(energy - nrg_old) < tol:
            if Params.output: print(f"AM Converged at iteration {niter+1}")
            break
        nrg_old = energy
    
    states_repro_MAM = V_reduced @ Q_reduced + V_reduced_nonlin @ Xi @ poly
    _report_reconstruction(Q_shifted, states_repro_MAM, V_X, V_T,
                           "Alternating minimization")

    return Q_reduced, V_reduced, V_reduced_nonlin, Xi

# The manuscript calls the two manifold variants MPOD and MAM, the code calls
# them NL-POD and AM. Both spellings are accepted here so that a name taken
# from the paper cannot silently miss every branch.
_BASIS_ALIASES = {
    'POD': 'POD',
    'POD_CNN': 'POD',
    'NL-POD': 'NL-POD',
    'MPOD': 'NL-POD',
    'AM': 'AM',
    'MAM': 'AM',
}


def get_basis_and_reduced_data(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states):
    """Entry point for basis selection based on configuration parameters.

    An unknown name raises. The previous version printed a message, and only
    with output enabled, then returned four Nones; the run continued and failed
    somewhere further down, on a shape or an attribute, with nothing left to
    point at the basis name that caused it.
    """
    requested = Params.basis
    kind = _BASIS_ALIASES.get(requested)

    if kind is None:
        raise ValueError(
            f"unknown basis '{requested}'; expected one of "
            f"{sorted(_BASIS_ALIASES)}"
        )

    if kind == 'POD':
        return _linear_pod_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states)
    if kind == 'NL-POD':
        return _nonlinear_pod_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states)
    return _nonlinear_basis(V_X, V_T, V_w1, V_w2, Q_shifted, reference_states)

def basis_multi(*args):
    """
    Construct a block-diagonal basis from multiple matrices and ranks.
    Input format: (Matrix1, Rank1, Matrix2, Rank2, ...)
    """
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be provided as pairs of (Matrix, Rank).")

    matrices, ranks = args[::2], args[1::2]
    reduced = [U[:, :r] for U, r in zip(matrices, ranks)]
    return la.block_diag(*reduced)

def basis_nonlin_multi(*args):
    """
    Construct a block-diagonal nonlinear basis extension.
    Input format: (Matrix1, RankLin1, RankNonlin1, ...)
    """
    if len(args) % 3 != 0:
        raise ValueError("Arguments must be provided as triplets of (Matrix, RankLin, RankNonlin).")

    matrices, r_lin, r_non = args[::3], args[1::3], args[2::3]
    reduced = [U[:, r:r+q] for U, r, q in zip(matrices, r_lin, r_non)]
    return la.block_diag(*reduced)

###############################################################################
# SVD AND ENERGY PLOTTING FUNCTIONS
###############################################################################

def cumulative_energy(singular_values, thresh=0.9999, ax=None, name_tag=None):
    """
    Plot and calculate the cumulative energy of singular values.

    Energy(j) = sum(sigma_{1:j}^2) / sum(sigma^2)

    The rank is the first index at which the cumulative energy reaches the
    threshold, so the energy actually retained is at or above it, sometimes far
    above. Both are reported, because the retained energy fixes the floor of
    the projection error at sqrt(1 - retained): the per-field reconstruction
    errors printed by the basis routines have to sit at or above that number,
    which makes the two printouts check each other.
    """
    svdvals2 = np.sort(singular_values)[::-1]**2
    cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)

    one_thresh = np.isscalar(thresh)
    if one_thresh: thresh = [thresh]

    # searchsorted returns len(cum_energy) for a threshold above every entry,
    # and the +1 would then index past the end of the spectrum. Rounding puts
    # the last entry marginally below a threshold of 1.0 often enough for this
    # to matter.
    n_modes = cum_energy.size
    ranks = [min(int(np.searchsorted(cum_energy, xi)) + 1, n_modes)
             for xi in thresh]

    if Params.output:
        retained = float(cum_energy[ranks[0] - 1])
        floor = float(np.sqrt(max(0.0, 1.0 - retained)))
        print(f"Rank r = {ranks[0]} required to exceed {thresh[0]*100:.2f}% "
              f"energy; retains {retained*100:.4f}%, so no projection onto "
              f"this basis can do better than {floor:.4%}.")

        plt.figure(figsize=(8, 6), dpi=300)
        ax = plt.subplot(111)
        j = np.arange(1, singular_values.size + 1)
        color = mpi_colors['mpi_red'] if 'temperature' in (name_tag or '').lower() else mpi_colors['mpi_green']
        marker = 'o' if 'temperature' in (name_tag or '').lower() else 'd'

        ax.plot(j, cum_energy*100, marker=marker, color=color, ms=12, lw=0, zorder=3)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        ax.set_xlim(-0.2, 30)
        ax.set_ylim(95.0, 100.2)
        ax.set_xlabel(r"Singular value index $j$", fontsize=20)
        ax.set_ylabel(r"Cumulative variance $\xi$ (%)", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()

        os.makedirs(FIGURE_DIR, exist_ok=True)
        file_name = f'cum_energy_{name_tag}.svg' if name_tag else 'cum_energy.svg'
        plt.savefig(os.path.join(FIGURE_DIR, file_name), bbox_inches='tight', transparent=True)
        plt.show()

    return ranks[0] if one_thresh else ranks

def svdval_decay(singular_values, tol=1e-8, normalize=True, plot=True, ax=None, name_tag=None):
    """Plot the decay of singular values relative to a tolerance threshold."""
    one_tol = np.isscalar(tol)
    if one_tol: tol = [tol]

    singular_values = np.sort(singular_values)[::-1]
    if normalize:
        singular_values /= singular_values[0]

    ranks = [np.count_nonzero(singular_values > epsilon) for epsilon in tol]

    if plot:
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, marker='o', lw=0, ms=12, color=mpi_colors['mpi_blue'], zorder=3)

        ax.set_xlim(1, 200)
        ax.set_xlabel(r"Singular value index $j$", fontsize=14)
        ax.set_ylabel(r"Relative singular values", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()

        os.makedirs(FIGURE_DIR, exist_ok=True)
        file_name = f'svd_decay_{name_tag}.svg' if name_tag else 'svd_decay.svg'
        plt.savefig(os.path.join(FIGURE_DIR, file_name), bbox_inches='tight', transparent=True)
        plt.show()

    return ranks[0] if one_tol else ranks

def residual_energy(singular_values, tol=1e-6, plot=True, ax=None):
    """
    Calculate the number of modes required for the residual energy to fall
    below a threshold. Residual = 1 - Cumulative Energy.
    """
    svdvals2 = np.sort(singular_values)[::-1] ** 2
    res_energy = 1 - (np.cumsum(svdvals2) / np.sum(svdvals2))

    one_tol = np.isscalar(tol)
    if one_tol: tol = [tol]
    ranks = [np.count_nonzero(res_energy > epsilon) + 1 for epsilon in tol]

    if plot:
        if ax is None: ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, res_energy, "C1.-", ms=10, lw=1, zorder=3)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Residual energy")

    return ranks[0] if one_tol else ranks

def svd_results(singular_values, name_tag):
    """
    Generate a combined plot showing both singular value decay and cumulative energy.
    Designed for high-quality publication output.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    ax1 = plt.subplot(111)

    svdvals2 = np.sort(singular_values)[::-1]**2
    cum_energy_percent = (np.cumsum(svdvals2) / np.sum(svdvals2)) * 100

    # Primary Axis: Singular Value Decay
    j = np.arange(1, singular_values.size + 1)
    ax1.semilogy(j, singular_values / singular_values[0], marker='o', ms=10, lw=0,
                 color=mpi_colors['mpi_blue'], zorder=3)
    ax1.set_xlim(1, 30)
    ax1.set_xlabel(r"Singular value index $j$", fontsize=24)
    ax1.set_ylabel(r"Relative singular values", fontsize=24)
    ax1.yaxis.label.set_color(mpi_colors['mpi_blue'])
    ax1.tick_params(axis='both', which='major', labelsize=24)

    # Secondary Axis: Cumulative Energy
    ax2 = ax1.twinx()
    ax2.plot(j, cum_energy_percent, marker='d', color=mpi_colors['mpi_green'], ms=10, lw=0, zorder=3)
    ax2.set_ylim(99.9, 100.01)
    ax2.set_ylabel(r"Cumulative energy $\xi$ (%)", fontsize=24)
    ax2.yaxis.label.set_color(mpi_colors['mpi_green'])
    ax2.tick_params(axis='both', which='major', labelsize=24)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    plt.title(f"SVD Analysis: {name_tag}" if name_tag else "SVD Analysis", fontsize=28)
    plt.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    file_name = f'combined_svd_{name_tag}.svg' if name_tag else 'combined_svd.svg'
    plt.savefig(os.path.join(FIGURE_DIR, file_name), bbox_inches='tight', transparent=True)
    plt.show()