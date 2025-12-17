import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qcf_connection_su3_embed import QCFConnectionSU3

def compute_gap(N=32, loop_size=6, samples=64, seed=1234):
    rng = np.random.default_rng(seed)
    conn = QCFConnectionSU3(N)

    scale = 10.0
    T_r = scale * rng.random((N, N))
    T_g = scale * rng.random((N, N))
    T_b = scale * rng.random((N, N))
    conn.seed_from_scalars(T_r, T_g, T_b)

    def square_loop(x, y, L):
        return [(x,y),(x+L,y),(x+L,y+L),(x,y+L),(x,y)]

    vals = []
    for _ in range(samples):
        x = rng.integers(loop_size + 2, N - loop_size - 2)
        y = rng.integers(loop_size + 2, N - loop_size - 2)
        loop = square_loop(x, y, loop_size)
        U = conn.transport(loop)
        vals.append(np.real(np.trace(U)) / 3.0)

    vals = np.array(vals)
    eigs = np.sort(np.abs(np.fft.fft(vals)))
    gap = np.log(eigs[-1] / eigs[-2])
    return gap

if __name__ == "__main__":
    gap = compute_gap()
    print("QCD-like mass gap =", gap)
    assert gap > 0.0
