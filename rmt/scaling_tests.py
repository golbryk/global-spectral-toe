import cupy as cp
from excitation_operator import build_excitation_operator
from ensemble_references import goe_eigenvalues, poisson_eigenvalues, r_statistic

cp.random.seed(123)

N_list = [200, 400, 800, 1200]
alpha_list = [0.15, 0.30, 0.60, 1.00]

print("N | alpha | r_TOE | r_GOE | r_Poisson")

for N in N_list:
    for alpha in alpha_list:
        H = build_excitation_operator(N, alpha)
        eig_toe = cp.sort(cp.linalg.eigvalsh(H))
        r_toe = r_statistic(eig_toe)

        r_goe = r_statistic(goe_eigenvalues(N))
        r_poi = r_statistic(poisson_eigenvalues(N))

        print(f"{N:4d} | {alpha:5.2f} | {r_toe:6.4f} | {r_goe:6.4f} | {r_poi:6.4f}")
