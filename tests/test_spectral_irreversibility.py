import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

if __name__ == "__main__":
    p = np.array([0.5, 0.3, 0.2])
    S0 = entropy(p)

    for _ in range(5):
        p = np.sort(p)[::-1]
        p = p / p.sum()

    S1 = entropy(p)

    print("Entropy initial =", S0)
    print("Entropy final   =", S1)

    assert S1 <= S0
