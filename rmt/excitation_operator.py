import cupy as cp

def build_relational_laplacian(N):
    L = cp.zeros((N, N), dtype=cp.float64)
    for i in range(N):
        L[i, i] = 2.0
        L[i, (i - 1) % N] = -1.0
        L[i, (i + 1) % N] = -1.0
    return L

def arnold_cat_phase(N, x0=0.1, y0=0.3):
    x = cp.zeros(N)
    y = cp.zeros(N)
    x[0] = x0
    y[0] = y0
    for n in range(1, N):
        x[n] = (2.0 * x[n-1] + y[n-1]) % 1.0
        y[n] = (x[n-1] + y[n-1]) % 1.0
    return cp.cos(2.0 * cp.pi * x)

def build_excitation_operator(N, alpha):
    L = build_relational_laplacian(N)
    phase = arnold_cat_phase(N)
    H = L + alpha * cp.diag(phase)
    return (H + H.T) / 2.0
