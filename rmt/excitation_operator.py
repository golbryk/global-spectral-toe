import cupy as cp
import numpy as np


def build_relational_laplacian(N):
    """1D ring Laplacian, fully vectorised on GPU."""
    idx = cp.arange(N, dtype=cp.int32)
    L = cp.zeros((N, N), dtype=cp.float64)
    L[idx, idx] = 2.0
    L[idx, (idx + 1) % N] = -1.0
    L[idx, (idx - 1) % N] = -1.0
    return L


def arnold_cat_phase(N, x0=0.1, y0=0.3):
    """
    Continuous Arnold cat map orbit on [0,1)^2.
    Map: (x,y) -> (2x+y mod 1, x+y mod 1)
    Phase at step n: cos(2*pi*x_n)

    The orbit is generated on CPU (pure Python: O(N) scalar ops, negligible vs
    GPU eigendecomposition at large N), then transferred to GPU in one batch.
    """
    xs = np.empty(N, dtype=np.float64)
    x, y = float(x0), float(y0)
    for n in range(N):
        xs[n] = x
        x, y = (2.0 * x + y) % 1.0, (x + y) % 1.0
    return cp.asarray(np.cos(2.0 * np.pi * xs))


def build_excitation_operator(N, alpha):
    """
    TOE excitation operator: L_ring + alpha * diag(cos(2pi*x_Arnold))
    Hermitian (real symmetric), built entirely on GPU.
    """
    L = build_relational_laplacian(N)
    phase = arnold_cat_phase(N)
    idx = cp.arange(N, dtype=cp.int32)
    L[idx, idx] += alpha * phase
    return L
