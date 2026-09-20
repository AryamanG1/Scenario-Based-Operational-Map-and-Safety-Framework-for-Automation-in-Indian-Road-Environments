"""Batched (GPU-capable) evaluation of the Stage 5 ODD copula density.

`odd_boundary.odd_density` evaluates phi_ODD(x) for ONE scene: five scalar
`gaussian_kde.evaluate` calls (each O(n) over the whole training sample), a
fresh `multivariate_normal.pdf` (which re-factorizes the correlation matrix
every call), and five `norm.ppf`s. Calling it once per row -- as
`fit_odd_copula`, `main.py`, `risk_estimation` and `feasibility_map` all did
-- is O(n^2 * d) with a ~1 ms Python/scipy floor per row. At the combined
IDD-20K-II + IDD117K corpus (~104k rows) that is ~5e10 kernel evaluations
plus 520k scipy calls, repeated four times per pipeline run.

This module computes the identical quantity for ALL rows at once, in torch,
in float64, chunked so peak memory stays bounded:

    empirical CDF       one torch.searchsorted per variable (rank/(n+1), clipped)
    norm.ppf            torch.special.ndtri
    copula density      closed-form Gaussian with the Cholesky factor of R
                        computed ONCE (scipy's mvn.pdf via eigendecomposition
                        gives the same number for a full-rank R)
    marginal KDE        (1/n) * sum_i exp(-(x - x_i)^2 / (2 h^2)) / sqrt(2 pi h^2)
                        with h^2 = kde.covariance, i.e. scipy's own bandwidth,
                        evaluated as a (chunk x n) reduction

The marginal-KDE term is the O(n^2) part and the one genuinely GPU-shaped
kernel in Stages 3-7: on a 40 GB A100-class card it takes seconds. On CPU
the same code is still ~100x faster than the per-row path because the work
is vectorized rather than interpreted.

Numerics are float64 throughout so results agree with the scipy path to
~1e-12 relative (see tests/odd/test_copula_gpu.py). Nothing here changes
the model: `ODDCopulaModel` is fitted, saved and loaded exactly as before.
"""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:  # pragma: no cover
    from src.odd.odd_boundary import ODDCopulaModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target bytes for one (chunk x n) float64 pairwise-difference tile. 256 MB
# keeps well inside any GPU while still amortizing kernel-launch overhead;
# on CPU it bounds transient RAM the same way.
_TILE_BYTES = 256 * 1024 * 1024

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _rows_per_chunk(n_samples: int) -> int:
    """Rows per chunk so a (chunk x n_samples) float64 tile is ~_TILE_BYTES."""
    return max(1, _TILE_BYTES // (8 * max(1, n_samples)))


def _as_matrix(model: "ODDCopulaModel", X: Union[pd.DataFrame, np.ndarray, Sequence[Dict]]) -> np.ndarray:
    """Coerces rows of ODD variables to a float64 (m, d) array in model order."""
    if isinstance(X, pd.DataFrame):
        return X[model.variables].to_numpy(dtype=np.float64)
    if isinstance(X, np.ndarray):
        arr = np.asarray(X, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    # Sequence of dicts (one per row), as odd_density takes.
    return np.array([[row[v] for v in model.variables] for row in X], dtype=np.float64)


def _kde_eval_batch(points: torch.Tensor, dataset: torch.Tensor, bandwidth_sq: float) -> torch.Tensor:
    """Gaussian KDE at `points` given a 1-D `dataset`, scipy-equivalent.

    Args:
        points: (m,) float64 on device.
        dataset: (n,) float64 on device.
        bandwidth_sq: `kde.covariance[0, 0]` (already includes Scott's factor^2).

    Returns:
        (m,) float64 density values.
    """
    n = dataset.shape[0]
    norm = 1.0 / (n * _SQRT_2PI * np.sqrt(bandwidth_sq))
    inv_2h2 = 0.5 / bandwidth_sq
    out = torch.empty_like(points)
    step = _rows_per_chunk(n)
    for start in range(0, points.shape[0], step):
        chunk = points[start : start + step]
        diff = chunk[:, None] - dataset[None, :]
        out[start : start + step] = torch.exp(-(diff * diff) * inv_2h2).sum(dim=1)
    return out * norm


def odd_density_batch(
    model: "ODDCopulaModel",
    X: Union[pd.DataFrame, np.ndarray, Sequence[Dict]],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Evaluates phi_ODD(x) = c(F(X1..Xd)) * prod f(Xi) for every row of X.

    Batched equivalent of `odd_boundary.odd_density`, same value per row.

    Args:
        model: A fitted ODDCopulaModel.
        X: Rows to evaluate: a DataFrame containing `model.variables`, an
            (m, d) array in `model.variables` order, or a sequence of dicts.
        device: Torch device; defaults to CUDA if available.

    Returns:
        A float64 array of shape (m,) of non-negative densities. Rows whose
        marginal-normal term underflows to zero return 0.0, as the scalar
        function does.
    """
    device = device or DEVICE
    Xnp = _as_matrix(model, X)
    m, d = Xnp.shape
    if m == 0:
        return np.zeros(0, dtype=np.float64)

    Xt = torch.from_numpy(np.ascontiguousarray(Xnp)).to(device)

    # --- Normal scores z = ndtri(F_hat(x)) ------------------------------
    z = torch.empty((m, d), dtype=torch.float64, device=device)
    marginal_f = torch.ones(m, dtype=torch.float64, device=device)
    for j, v in enumerate(model.variables):
        sorted_t = torch.from_numpy(np.ascontiguousarray(model.sorted_samples[v], dtype=np.float64)).to(device)
        n = sorted_t.shape[0]
        rank = torch.searchsorted(sorted_t, Xt[:, j].contiguous(), right=True).to(torch.float64)
        u = torch.clamp(rank / (n + 1), 1e-6, 1 - 1e-6)
        z[:, j] = torch.special.ndtri(u)

        kde = model.marginal_kdes[v]
        # np.array(copy=True): kde.dataset is a read-only view, which torch warns about.
        dataset = torch.from_numpy(np.array(kde.dataset[0], dtype=np.float64, copy=True)).to(device)
        marginal_f *= _kde_eval_batch(Xt[:, j], dataset, float(kde.covariance[0, 0]))

    # --- Gaussian copula density c(u) = phi_R(z) / prod phi(z_j) ---------
    R = torch.from_numpy(np.asarray(model.correlation_matrix, dtype=np.float64)).to(device)
    L = torch.linalg.cholesky(R)
    # solve L y = z^T  ->  z^T R^{-1} z = ||y||^2
    y = torch.linalg.solve_triangular(L, z.T, upper=False)
    quad = (y * y).sum(dim=0)
    log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
    log_mvn = -0.5 * quad - 0.5 * log_det - 0.5 * d * np.log(2.0 * np.pi)
    log_marg_norm = (-0.5 * z * z - 0.5 * np.log(2.0 * np.pi)).sum(dim=1)

    copula_density = torch.exp(log_mvn)
    marginal_normal = torch.exp(log_marg_norm)
    # Match the scalar function: it divides pdf by prod(norm.pdf(z)) and
    # returns 0.0 where that product underflows to <= 0.
    copula_c = torch.where(marginal_normal > 0, copula_density / marginal_normal, torch.zeros_like(copula_density))

    return (copula_c * marginal_f).cpu().numpy()


def classify_odd_region_batch(
    model: "ODDCopulaModel",
    X: Union[pd.DataFrame, np.ndarray, Sequence[Dict]],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Batched `odd_boundary.classify_odd_region`: one label per row.

    Args:
        model: A fitted ODDCopulaModel (with density_percentiles populated).
        X: Rows to classify (see `odd_density_batch`).
        device: Torch device; defaults to CUDA if available.

    Returns:
        An object array of "within" / "near" / "outside" strings.
    """
    from src.odd.odd_boundary import density_cutpoints

    p15, p50 = density_cutpoints(model)
    density = odd_density_batch(model, X, device=device)
    out = np.where(density < p15, "outside", np.where(density < p50, "near", "within"))
    return out.astype(object)
