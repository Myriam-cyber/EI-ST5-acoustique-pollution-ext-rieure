# -*- coding: utf-8 -*-
"""
multilevel_sweep.py

Balayage en fréquence (100–1000 Hz) pour comparer, pour chaque niveau de fractal :
- énergie après optimisation (χ_opt binaire),
- énergie pour mur totalement absorbant (χ_full = 1 sur la Robin).

Ce script importe :
    - multilevel.py (ton gros script d'optimisation)
    - les données matériau (MELAMINE)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import _env
import preprocessing
import processing

import multilevel  # ton script principal


# -------------------------------------------------------------------
# 1) Utilitaires : source "autoroute"
# -------------------------------------------------------------------
def build_road_source(M, N, spacestep):
    """Recrée la même excitation 'autoroute' que dans multilevel.py"""
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    f[:, :] = 0.0
    f_neu[:, :] = 0.0
    f_rob[:, :] = 0.0
    f_dir[:, :] = 0.0

    i_source = 0
    n_sources = 6
    spacing = N // (n_sources + 1)

    # sources réparties (voitures)
    for n in range(n_sources):
        j_src = (n + 1) * spacing
        if j_src < N:
            f_dir[i_source, j_src] = 2.0 + 0.0j

    # profil gaussien (plus fort au centre)
    for j in range(N):
        dist_center = abs(j - N // 2)
        f_dir[i_source, j] *= np.exp(-(dist_center / (N / 3.0)) ** 2)

    return f, f_dir, f_neu, f_rob


# -------------------------------------------------------------------
# 2) Calcul d'énergie pour une distribution χ donnée
# -------------------------------------------------------------------
def compute_energy_for_chi(domain_omega, chi, frequencies,
                           freq_tab_alpha, alpha_tab_alpha,
                           c, L_ref):
    """Calcule l'énergie acoustique J(f) pour une distribution χ donnée"""
    (M, N) = domain_omega.shape
    spacestep = 1.0 / N

    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob_dummy, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)

    f, f_dir, f_neu, f_rob = build_road_source(M, N, spacestep)

    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    energies = np.zeros_like(frequencies, dtype=float)

    for kf, f_Hz in enumerate(frequencies):
        k_phys = 2.0 * np.pi * f_Hz / c
        omega = k_phys * L_ref

        Alpha_f = multilevel.alpha_of_freq(f_Hz, freq_tab_alpha, alpha_tab_alpha)
        alpha_rob = Alpha_f * chi

        u = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob
        )

        J = np.sum(np.abs(u[mask_dom]) ** 2) * (spacestep ** 2)
        energies[kf] = float(np.real(J))

        if (kf == 0) or (kf == len(frequencies) - 1) or ((kf + 1) % 10 == 0):
            print(f"    f = {f_Hz:.1f} Hz → J = {energies[kf]:.6e}")

    return energies


# -------------------------------------------------------------------
# 3) Main : balayage en fréquence (après opt vs full absorbant)
# -------------------------------------------------------------------
if __name__ == "__main__":

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  FREQUENCY SWEEP OF OPTIMIZED FRACTAL WALLS                  ║
    ║  (Optimized χ_opt vs Fully Absorbent χ_full)                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # ------------------ paramètres globaux -------------------------
    N = 50
    f_ref = 180.0
    V_obj = 0.55
    zeta0 = 0.15
    mu1 = 1e-9
    max_iter = 300
    delta = 1e-4

    c = 343.0
    L_ref = 1.0

    f_min, f_max = 100.0, 1000.0
    n_f = 46
    frequencies = np.linspace(f_min, f_max, n_f)

    levels_to_test = [0, 1, 2, 3]

    print("Frequency sweep parameters:")
    print(f"  f in [{f_min:.1f}, {f_max:.1f}] Hz, Npts = {n_f}")
    print(f"  Levels: {levels_to_test}")
    print(f"  β target = {V_obj:.2%}\n")

    root_out = "freq_sweep_results"
    os.makedirs(root_out, exist_ok=True)

    freq_tab_alpha, alpha_tab_alpha = multilevel.load_alpha_table(
        "dta_freq_MELAMINE.mtx",
        "dta_alpha_MELAMINE.mtx"
    )

    all_results = {}

    # ------------------ boucle sur les niveaux ---------------------
    for level in levels_to_test:
        print("\n" + "#" * 70)
        print(f"# LEVEL {level}")
        print("#" * 70 + "\n")

        print(f">> Optimizing at reference frequency {f_ref:.1f} Hz to get χ_opt\n")

        res_opt = multilevel.run_optimization_for_level(
            level=level,
            N=N,
            f_Hz=f_ref,
            V_obj=V_obj,
            zeta0=zeta0,
            mu1=mu1,
            max_iter=max_iter,
            delta=delta
        )

        domain_omega = res_opt["domain"]
        chi_opt = res_opt["chi_bin"]  # après optimisation (binaire)
        (M, N_eff) = domain_omega.shape

        mask_R = (domain_omega == _env.NODE_ROBIN)
        chi_full = np.zeros_like(chi_opt)
        chi_full[mask_R] = 1.0

        out_level = os.path.join(root_out, f"level_{level}")
        os.makedirs(out_level, exist_ok=True)

        print("\n--- Sweeping frequencies for χ_opt (optimized wall) ---")
        energies_opt = compute_energy_for_chi(
            domain_omega, chi_opt, frequencies,
            freq_tab_alpha, alpha_tab_alpha,
            c, L_ref
        )

        print("\n--- Sweeping frequencies for χ_full (fully absorbent wall) ---")
        energies_full = compute_energy_for_chi(
            domain_omega, chi_full, frequencies,
            freq_tab_alpha, alpha_tab_alpha,
            c, L_ref
        )

        # Sauvegarde
        np.savetxt(os.path.join(out_level, "frequencies_Hz.txt"), frequencies)
        np.savetxt(os.path.join(out_level, "energy_opt.txt"), energies_opt)
        np.savetxt(os.path.join(out_level, "energy_full.txt"), energies_full)

        # Figure comparaison
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, energies_opt, "-s", linewidth=1.8,
                 label="Après optimisation (χ_opt binaire)")
        plt.plot(frequencies, energies_full, "-^", linewidth=1.8,
                 label="Mur totalement absorbant (χ_full)")

        plt.xlabel("Fréquence (Hz)", fontsize=12)
        plt.ylabel("Énergie acoustique J", fontsize=12)
        plt.title(f"Niveau fractal {level} – χ_opt vs χ_full",
                  fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        fig_name = os.path.join(out_level, f"energy_comparison_level_{level}.png")
        plt.savefig(fig_name, dpi=300)
        plt.close()

        print(f"\nSaved figure: {fig_name}")

        # Sauvegarde des données pour usage global
        all_results[level] = {
            "frequencies": frequencies,
            "energy_opt": energies_opt,
            "energy_full": energies_full
        }

    # -------------------------------------------------------------------
    # 4) Comparaison globale entre niveaux (optimisés uniquement)
    # -------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for level in levels_to_test:
        freqs = all_results[level]["frequencies"]
        plt.plot(freqs, all_results[level]["energy_opt"], lw=2, label=f"Niveau {level}")
    plt.xlabel("Fréquence (Hz)", fontsize=12)
    plt.ylabel("Énergie acoustique J", fontsize=12)
    plt.title("Énergie optimisée (χ_opt) selon la fréquence et le niveau fractal",
              fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(root_out, "energy_all_levels_optimized.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("Frequency sweep completed for all levels.")
    print(f"Results and figures saved under: {root_out}/")
    print("=" * 70 + "\n")
