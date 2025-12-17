import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qcf_core import spectral_evolution

def test_arrow():
    eigs = np.array([0.35, 0.12, 0.04])
    amps_fwd = spectral_evolution(eigs, steps=8, forward=True)
    amps_bwd = spectral_evolution(eigs, steps=8, forward=False)

    assert amps_fwd[-1] < amps_fwd[0]
    assert amps_bwd[-1] > amps_bwd[0]

    print("Forward decay OK")
    print("Backward blow-up OK")

if __name__ == "__main__":
    test_arrow()
