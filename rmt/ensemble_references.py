import cupy as cp

def goe_eigenvalues(N):
    A = cp.random.normal(0.0, 1.0, (N, N))
    H = (A + A.T) / cp.sqrt(2.0 * N)
    return cp.sort(cp.linalg.eigvalsh(H))

def poisson_eigenvalues(N):
    eig = cp.cumsum(cp.random.exponential(1.0, N))
    return cp.sort(eig)

def r_statistic(eigs):
    s = eigs[1:] - eigs[:-1]
    s = s / cp.mean(s)
    r = cp.minimum(s[:-1], s[1:]) / cp.maximum(s[:-1], s[1:])
    r = r[cp.isfinite(r)]
    return float(cp.mean(r))
