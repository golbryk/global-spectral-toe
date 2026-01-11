import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================

LOG_DIR = "logs"
FIG_DIR = "../theory/figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Parse logs
# ============================================================

pattern = re.compile(
    r"\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
)

data = []

for fname in os.listdir(LOG_DIR):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(LOG_DIR, fname), "r") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                N = int(m.group(1))
                alpha = float(m.group(2))
                r_toe = float(m.group(3))
                r_goe = float(m.group(4))
                r_poi = float(m.group(5))
                data.append((N, alpha, r_toe, r_goe, r_poi))

if not data:
    raise RuntimeError("No valid RMT data found in logs/")

# ============================================================
# Organize data
# ============================================================

by_N = defaultdict(list)
by_alpha = defaultdict(list)

for N, alpha, r_toe, r_goe, r_poi in data:
    by_N[N].append((alpha, r_toe, r_goe, r_poi))
    by_alpha[alpha].append((N, r_toe, r_goe, r_poi))

# ============================================================
# Figure 1: r vs N (for fixed alpha values)
# ============================================================

plt.figure(figsize=(7, 5))

for alpha in sorted(by_alpha.keys()):
    Ns = []
    rvals = []
    for N, r_toe, _, _ in sorted(by_alpha[alpha]):
        Ns.append(N)
        rvals.append(r_toe)
    plt.plot(Ns, rvals, marker="o", label=f"TOE alpha={alpha}")

# Reference bands
plt.axhline(0.5359, linestyle="--", color="black", label="GOE")
plt.axhline(0.3863, linestyle=":", color="black", label="Poisson")

plt.xlabel("Matrix size N")
plt.ylabel("r-statistic")
plt.title("RMT scaling with matrix size")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "rmt_scaling_r_vs_N.png"))
plt.close()

# ============================================================
# Figure 2: r vs alpha (for fixed N values)
# ============================================================

plt.figure(figsize=(7, 5))

for N in sorted(by_N.keys()):
    alphas = []
    rvals = []
    for alpha, r_toe, _, _ in sorted(by_N[N]):
        alphas.append(alpha)
        rvals.append(r_toe)
    plt.plot(alphas, rvals, marker="o", label=f"TOE N={N}")

plt.axhline(0.5359, linestyle="--", color="black", label="GOE")
plt.axhline(0.3863, linestyle=":", color="black", label="Poisson")

plt.xlabel("Coupling alpha")
plt.ylabel("r-statistic")
plt.title("RMT scaling with coupling strength")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "rmt_scaling_r_vs_alpha.png"))
plt.close()

# ============================================================
# Figure 3: Scatter overview (all points)
# ============================================================

plt.figure(figsize=(7, 5))

for N, alpha, r_toe, _, _ in data:
    plt.scatter(alpha, r_toe, color="black", s=20)

plt.axhline(0.5359, linestyle="--", color="black", label="GOE")
plt.axhline(0.3863, linestyle=":", color="black", label="Poisson")

plt.xlabel("Coupling alpha")
plt.ylabel("r-statistic")
plt.title("All RMT results (no selection)")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "rmt_all_points.png"))
plt.close()

print("Figures generated in theory/figures/")
