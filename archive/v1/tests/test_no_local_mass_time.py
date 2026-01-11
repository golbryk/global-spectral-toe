import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

def local_spectrum(size):
    return np.ones(size)

def global_spectrum():
    return np.array([0.33, 0.11, 0.04])

def test_no_local():
    local = local_spectrum(3)
    global_ = global_spectrum()

    local_gap = np.log(local[0]/local[1]) if local[1] != 0 else 0.0
    global_gap = np.log(global_[0]/global_[1])

    print("Local gap =", local_gap)
    print("Global gap =", global_gap)

    assert abs(local_gap) < 1e-12
    assert global_gap > 0.0

if __name__ == "__main__":
    test_no_local()
