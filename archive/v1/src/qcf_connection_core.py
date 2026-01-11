import cupy as cp

# ============================================================
# SU(2) CONNECTION CORE — PURE GAUGE FIELD
# ============================================================

class QCFConnectionSU2:
    def __init__(self, N):
        self.N = N
        self.shape = (N, N, N)
        self.Ax = cp.zeros(self.shape, dtype=cp.float32)
        self.Ay = cp.zeros(self.shape, dtype=cp.float32)

    def seed_radial_connection(self, strength=0.01):
        N = self.N
        cx = cy = N // 2
        x = cp.arange(N, dtype=cp.float32) - cx
        y = cp.arange(N, dtype=cp.float32) - cy
        X, Y = cp.meshgrid(x, y, indexing='ij')
        R = cp.sqrt(X*X + Y*Y) + 1e-6
        for z in range(N):
            self.Ax[:, :, z] = (-Y / R) * strength
            self.Ay[:, :, z] = ( X / R) * strength


    def seed_from_scalar(self, T, alpha=0.01):
        """
        Initialize SU(2) connection from scalar causal time T(x,y)
        by taking its gradient as a pure gauge field.
        """
        import cupy as cp

        T = cp.asarray(T, dtype=cp.float32)

        dTx = cp.zeros_like(T)
        dTy = cp.zeros_like(T)

        dTx[1:-1, :] = T[2:, :] - T[:-2, :]
        dTy[:, 1:-1] = T[:, 2:] - T[:, :-2]

        for z in range(self.N):
            self.Ax[:, :, z] = alpha * dTx
            self.Ay[:, :, z] = alpha * dTy

    def update_from_field(self, mx, my, alpha=0.01):
        if mx is None or my is None:
            return
        self.Ax += alpha * my
        self.Ay -= alpha * mx

    def norm(self):
        return float(cp.sqrt(cp.mean(self.Ax**2 + self.Ay**2)).get())

    def line_integral(self, pts):
        acc = 0.0
        for (x0, y0, z0), (x1, y1, z1) in zip(pts[:-1], pts[1:]):
            dx = x1 - x0
            dy = y1 - y0
            acc += float(self.Ax[x0, y0, z0].get() * dx + self.Ay[x0, y0, z0].get() * dy)
        return acc
    
    def transport_step(self, x0, y0, x1, y1, z=0):
        """
        SU(2) parallel transport for a single lattice step.
        Returns a 2x2 SU(2) matrix.
        """
        import cupy as cp
        import numpy as np

        dx = x1 - x0
        dy = y1 - y0

        # line integral A · dx
        theta = float(
            self.Ax[x0, y0, z].get() * dx +
            self.Ay[x0, y0, z].get() * dy
        )

        # SU(2) generator sigma_y (minimal nontrivial choice)
        U = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=complex)

        return U

