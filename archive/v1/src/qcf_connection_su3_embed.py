import numpy as np
from qcf_connection_core import QCFConnectionSU2

class QCFConnectionSU3:
    """
    SU(3) connection built from overlapping SU(2) connections
    """

    def __init__(self, N):
        self.N = N
        self.su2_rg = None
        self.su2_gb = None
        self.su2_br = None

    def seed_from_scalars(self, T_r, T_g, T_b):
        self.su2_rg = QCFConnectionSU2(self.N)
        self.su2_gb = QCFConnectionSU2(self.N)
        self.su2_br = QCFConnectionSU2(self.N)

        self.su2_rg.seed_from_scalar(T_r - T_g)
        self.su2_gb.seed_from_scalar(T_g - T_b)
        self.su2_br.seed_from_scalar(T_b - T_r)



    def transport(self, path):
        U = np.eye(3, dtype=complex)

        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):

            U_rg = self.su2_rg.transport_step(x0, y0, x1, y1)
            U_gb = self.su2_gb.transport_step(x0, y0, x1, y1)
            U_br = self.su2_br.transport_step(x0, y0, x1, y1)

            M = np.eye(3, dtype=complex)
            M[np.ix_([0,1],[0,1])] = U_rg
            M[np.ix_([1,2],[1,2])] = U_gb
            M[np.ix_([2,0],[2,0])] = U_br

            U = M @ U

        return U
