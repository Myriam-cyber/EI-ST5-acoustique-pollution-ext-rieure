# -*- coding: utf-8 -*-
"""
Frequency sweep of acoustic energy
==================================

Pour chaque niveau de fractal et pour plusieurs β (fraction de surface
absorbante), on :

  1) optimise χ à une fréquence de référence f_opt (via multilevel.py),
  2) garde χ_opt (binaire),
  3) balaye les fréquences de f_min à f_max et on calcule l'énergie J(f).

On trace ensuite J(f) pour :
   - le mur totalement absorbant (χ = 1 sur Robin),
   - la paroi optimisée pour chaque β.

Les figures sont stockées dans le dossier 'freq_scan_results'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import _env
import preprocessing
import processing

import multilevel  # le fichier précédent


# -------------------------------------------------------------------
# 1) Calcul de l'énergie pour un χ donné, en fonction de f
# -------------------------------------------------------------------
def compute_energy_for_chi(domain_omega, chi, frequencies,
                           freq_tab_alpha, alpha_tab_alpha,
                           c=343.0, L_ref=1.0,
                           amplitude=2.0, n_sources=6):
    """
    Calcule J(f) = ∫_Ω |u|² dx pour un χ fixé et un balayage en fréquence.

    - domain_omega : géométrie,
    - chi          : distribution (0/1) sur le domaine,
    - frequencies  : tableau des fréquences (Hz),
    - freq_tab_alpha, alpha_tab_alpha : tables pour α(f) (multilevel.load_alpha_table),
    - amplitude, n_sources : paramètres de la source "autoroute".
    """
    M, N = domain_omega.shape
    spacestep = 1.0 / N

    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    energies = np.zeros_like(frequencies, dtype=float)

    for kf, f_Hz in enumerate(frequencies):
        # Coefficients PDE (indépendants de f sauf α_rob, qu'on redéfinit ensuite)
        beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob_dummy, beta_rob = \
            preprocessing._set_coefficients_of_pde(M, N)

        # RHS + source autoroute
        f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
        f, f_dir, f_neu, f_rob = multilevel.build_road_source(
            f, f_dir, f_neu, f_rob,
            amplitude=amplitude, n_sources=n_sources
        )

        # Nombre d'onde
        k_phys = 2.0 * np.pi * f_Hz / c
        omega = k_phys * L_ref

        # Alpha(f) à cette fréquence
        Alpha_f = multilevel.alpha_of_freq(f_Hz, freq_tab_alpha, alpha_tab_alpha)
        alpha_rob = Alpha_f * chi

        # Résolution Helmholtz
        u = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, alpha_rob
        )

        # Énergie
        J = np.sum(np.abs(u[mask_dom]) ** 2) * (spacestep ** 2)
        energies[kf] = float(np.real(J))

        if (kf == 0) or (kf == len(frequencies) - 1) or ((kf + 1) % 10 == 0):
            print(f"    f = {f_Hz:.1f} Hz → J = {energies[kf]:.6e}")

    return energies


if __name__ == '__main__':

    # ----------------------------------------------------------------
    # PARAMÈTRES GLOBAUX
    # ----------------------------------------------------------------
    N = 50
    levels_to_test = [0, 1, 2, 3]      # niveaux de fractal
    beta_list = [0.4,0.5,0.6, 0.8]  # fractions de surface absorbante

    f_opt = 180.0       # fréquence de référence pour l'optimisation de χ
    f_min = 100.0
    f_max = 400.0
    n_freq = 200
    frequencies = np.linspace(f_min, f_max, n_freq)

    zeta0 = 0.15
    mu1 = 1e-9
    max_iter = 200       # tu peux réduire pour que ce soit moins long
    delta = 1e-4

    c = 343.0
    L_ref = 1.0

    amplitude_source = 2.0
    n_sources = 6

    # dossier de sortie
    out_dir = 'freq_scan_results'
    os.makedirs(out_dir, exist_ok=True)

    print("\n===============================================")
    print(" FREQUENCY SWEEP WITH OPTIMIZED χ (multilevel) ")
    print("===============================================\n")
    print(f"Grid: N = {N}, levels = {levels_to_test}")
    print(f"Betas: {beta_list}")
    print(f"Frequency sweep: [{f_min}, {f_max}] Hz, n = {n_freq}")
    print(f"Optimization frequency: f_opt = {f_opt} Hz\n")

    # Tables α(f)
    freq_tab_alpha, alpha_tab_alpha = multilevel.load_alpha_table(
        'dta_freq_MELAMINE.mtx',
        'dta_alpha_MELAMINE.mtx'
    )

    # Pour sauver les résultats numériques
    all_results = {}

    for level in levels_to_test:
        print("\n" + "#" * 72)
        print(f"# LEVEL {level} - frequency sweep")
        print("#" * 72 + "\n")

        level_results = {}

        # On optimise χ pour chaque β à f_opt, puis on garde χ_bin
        for beta in beta_list:
            print("\n" + "-" * 60)
            print(f"Optimizing χ for level {level}, β = {beta:.2f} at f_opt = {f_opt} Hz")
            print("-" * 60 + "\n")

            res_opt = multilevel.run_optimization_for_level(
                level=level,
                N=N,
                f_Hz=f_opt,
                V_obj=beta,
                zeta0=zeta0,
                mu1=mu1,
                max_iter=max_iter,
                delta=delta
            )

            domain = res_opt['domain']
            chi_bin = res_opt['chi_bin']

            print("\nComputing energy vs frequency for this optimized χ...")
            energies_beta = compute_energy_for_chi(
                domain_omega=domain,
                chi=chi_bin,
                frequencies=frequencies,
                freq_tab_alpha=freq_tab_alpha,
                alpha_tab_alpha=alpha_tab_alpha,
                c=c, L_ref=L_ref,
                amplitude=amplitude_source,
                n_sources=n_sources
            )

            level_results[f'beta_{beta:.2f}'] = {
                'beta': beta,
                'chi_bin': chi_bin,
                'energies': energies_beta
            }

        # Cas paroi totalement absorbante (χ = 1 sur Robin)
        print("\n" + "-" * 60)
        print(f"Computing energy vs frequency for FULLY ABSORBENT wall (level {level})")
        print("-" * 60 + "\n")

        # Géométrie (on peut réutiliser domain du dernier res_opt)
        domain = res_opt['domain']
        M, N_ = domain.shape
        chi_full = np.ones((M, N_), dtype=float)
        chi_full = preprocessing.set2zero(chi_full, domain)  # 0 hors Robin/Ω

        energies_full = compute_energy_for_chi(
            domain_omega=domain,
            chi=chi_full,
            frequencies=frequencies,
            freq_tab_alpha=freq_tab_alpha,
            alpha_tab_alpha=alpha_tab_alpha,
            c=c, L_ref=L_ref,
            amplitude=amplitude_source,
            n_sources=n_sources
        )

        level_results['full_absorbent'] = {
            'beta': 1.0,
            'chi_full': chi_full,
            'energies': energies_full
        }

        all_results[f'level_{level}'] = {
            'frequencies': frequencies,
            'results': level_results
        }

        # ----------------------------------------------------------------
        # PLOT pour ce niveau : courbes J(f) pour tous les β + fully absorbent
        # ----------------------------------------------------------------
        plt.figure(figsize=(10, 6))

        # paroi totalement absorbante
        plt.plot(frequencies, energies_full,
                 'k--', linewidth=2, label='Fully absorbent (β=1)')

        # pour chaque β optimisé
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        for k, beta in enumerate(beta_list):
            key = f'beta_{beta:.2f}'
            if key in level_results:
                E_beta = level_results[key]['energies']
                plt.plot(frequencies, E_beta,
                         '-', linewidth=2,
                         color=colors[k % len(colors)],
                         label=f'Optimized β={beta:.2f}')

        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Acoustic energy J', fontsize=12)
        plt.title(f'Level {level}: Energy vs Frequency after optimization', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'level_{level}_energy_vs_frequency.png'),
                    dpi=200)
        plt.close()

    # --------------------------------------------------------------------
    # Sauvegarde d’un petit résumé texte
    # --------------------------------------------------------------------
    summary_file = os.path.join(out_dir, 'freq_scan_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("FREQUENCY SWEEP WITH OPTIMIZED χ\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Levels: {levels_to_test}\n")
        f.write(f"Betas: {beta_list}\n")
        f.write(f"Frequency range: [{f_min}, {f_max}] Hz, n = {n_freq}\n")
        f.write(f"Optimization frequency: f_opt = {f_opt} Hz\n\n")

        for level in levels_to_test:
            key_level = f'level_{level}'
            f.write("-" * 70 + "\n")
            f.write(f"Level {level}\n")
            f.write("-" * 70 + "\n")
            res_level = all_results[key_level]['results']

            # fully absorbent
            E_full = res_level['full_absorbent']['energies']
            f.write(f"  FULL ABSORBENT: J_min = {E_full.min():.6e}, "
                    f"J_max = {E_full.max():.6e}\n")

            for beta in beta_list:
                key_beta = f'beta_{beta:.2f}'
                if key_beta in res_level:
                    E_beta = res_level[key_beta]['energies']
                    f.write(f"  β={beta:.2f}: J_min = {E_beta.min():.6e}, "
                            f"J_max = {E_beta.max():.6e}\n")
            f.write("\n")

    print(f"\nAll frequency-sweep plots and summary saved in '{out_dir}'\n")
