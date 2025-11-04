# -*- coding: utf-8 -*-
"""
Frequency sweep for multilevel acoustic optimization
====================================================

Ce script :
- importe le module multilevel.py
- fixe un volume fraction β (V_obj)
- balaye la fréquence jusqu'à 1000 Hz
- pour chaque niveau de fractale, calcule l'énergie après optimisation
- produit des figures dans un nouveau dossier: freq_sweep_results/
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import multilevel  # <-- ton fichier précédent, nommé multilevel.py


def run_frequency_sweep(levels_to_test,
                        N,
                        V_obj,
                        f_min=100.0,
                        f_max=1000.0,
                        n_freq=25,
                        zeta0=0.15,
                        mu1=1e-9,
                        max_iter=150,
                        delta=1e-4):
    """
    Balayage fréquentiel pour plusieurs niveaux de fractale.

    Parameters
    ----------
    levels_to_test : list[int]
        Liste des niveaux de fractale (ex: [0,1,2,3])
    N : int
        Résolution spatiale (même que dans multilevel)
    V_obj : float
        Volume fraction β (0 < β < 1)
    f_min, f_max : float
        Bornes de fréquences (en Hz)
    n_freq : int
        Nombre de points de fréquence
    zeta0 : float
        Pas de gradient de base (sera adapté par multilevel.run_optimization_for_level)
    mu1 : float
        Pénalité de volume
    max_iter : int
        Nombre max d'itérations d'optimisation pour chaque (niveau, fréquence)
    delta : float
        Tolérance de convergence sur ||Δχ||∞

    Returns
    -------
    results : dict
        Dictionnaire contenant J0 et J_bin pour chaque niveau et fréquence.
    """

    # Dossier de sortie
    output_dir = "freq_sweep_results"
    os.makedirs(output_dir, exist_ok=True)

    # Discrétisation des fréquences
    frequencies = np.linspace(f_min, f_max, n_freq)

    # Structure de stockage :
    # results[level]["freqs"], ["J0"], ["J_bin"]
    results = {}

    print("\n" + "#" * 70)
    print("# FREQUENCY SWEEP")
    print("# Levels:", levels_to_test)
    print("# Frequencies: from %.1f Hz to %.1f Hz (%d points)" % (f_min, f_max, n_freq))
    print("# Target β = %.2f" % V_obj)
    print("#" * 70 + "\n")

    for level in levels_to_test:
        print("\n" + "=" * 70)
        print(f"STARTING SWEEP FOR LEVEL {level}")
        print("=" * 70 + "\n")

        J0_list = []
        Jbin_list = []

        for i, f_Hz in enumerate(frequencies):
            print("\n" + "-" * 60)
            print(f"[Level {level}] Frequency {i+1}/{n_freq}: f = {f_Hz:.1f} Hz")
            print("-" * 60)

            try:
                res = multilevel.run_optimization_for_level(
                    level=level,
                    N=N,
                    f_Hz=f_Hz,
                    V_obj=V_obj,
                    zeta0=zeta0,
                    mu1=mu1,
                    max_iter=max_iter,
                    delta=delta
                )
                J0_list.append(res["J0"])
                Jbin_list.append(res["J_bin"])

                print(f"  -> J0   = {res['J0']:.6e}")
                print(f"  -> Jbin = {res['J_bin']:.6e}")
                print(f"  -> ΔJ/J0 = {(res['J0'] - res['J_bin'])/res['J0']:.2%}")

            except Exception as e:
                print(f"  ERROR for level {level}, f = {f_Hz:.1f} Hz: {e}")
                J0_list.append(np.nan)
                Jbin_list.append(np.nan)

        J0_arr = np.array(J0_list)
        Jbin_arr = np.array(Jbin_list)

        results[level] = {
            "frequencies": frequencies,
            "J0": J0_arr,
            "J_bin": Jbin_arr,
        }

        # Sauvegarde des courbes pour ce niveau
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, J0_arr, "k--", lw=1.5, label="Uncontrolled J₀")
        plt.plot(frequencies, Jbin_arr, "b-", lw=2.0, label="Optimized J_opt")
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("Energy J", fontsize=12)
        plt.title(f"Level {level}: Energy vs Frequency (β = {V_obj:.2f})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(output_dir, f"energy_level_{level}.png")
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"\n[Level {level}] Saved per-level sweep figure: {filename}")

    # ==== Figures globales (tous niveaux) ====

    # 1) J_bin pour tous les niveaux sur un même graphe
    plt.figure(figsize=(10, 6))
    for level in levels_to_test:
        freqs = results[level]["frequencies"]
        Jbin = results[level]["J_bin"]
        plt.plot(freqs, Jbin, lw=2, label=f"Level {level}")
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Optimized Energy J_opt", fontsize=12)
    plt.title(f"Optimized Energy vs Frequency for all levels (β = {V_obj:.2f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, "energy_all_levels_optimized.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"\nSaved global optimized energy plot: {filename}")

    # 2) J0 (non contrôlé) pour tous les niveaux
    plt.figure(figsize=(10, 6))
    for level in levels_to_test:
        freqs = results[level]["frequencies"]
        J0 = results[level]["J0"]
        plt.plot(freqs, J0, lw=1.5, linestyle="--", label=f"Level {level}")
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Uncontrolled Energy J₀", fontsize=12)
    plt.title("Uncontrolled Energy vs Frequency for all levels", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, "energy_all_levels_uncontrolled.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved global uncontrolled energy plot: {filename}")

    # 3) Sauvegarde des données numériques
    np.savez(
        os.path.join(output_dir, "freq_sweep_data.npz"),
        levels=np.array(levels_to_test),
        frequencies=frequencies,
        **{f"J0_level_{lvl}": results[lvl]["J0"] for lvl in levels_to_test},
        **{f"Jbin_level_{lvl}": results[lvl]["J_bin"] for lvl in levels_to_test},
    )
    print(f"\nSaved numerical data: {os.path.join(output_dir, 'freq_sweep_data.npz')}")

    return results


if __name__ == "__main__":

    # Même type de paramètres globaux que dans multilevel.py
    N = 50               # grid
    V_obj = 0.4         # volume fraction β (même que dans ton main actuel)
    f_min = 100.0
    f_max = 1000.0
    n_freq = 25          # à ajuster selon le temps de calcul que tu acceptes
    zeta0 = 0.15
    mu1 = 1e-9
    max_iter = 120       # plus petit que 300 pour ne pas exploser le temps
    delta = 1e-4

    levels_to_test = [0, 1, 2, 3]

    print("\n" + "="*70)
    print("FREQUENCY SWEEP MULTI-LEVEL")
    print("="*70)
    print(f"  Levels:            {levels_to_test}")
    print(f"  Frequency range:   [{f_min:.1f}, {f_max:.1f}] Hz, n = {n_freq}")
    print(f"  Grid resolution:   N = {N}")
    print(f"  Target β:          {V_obj:.2f}")
    print(f"  max_iter:          {max_iter}")
    print(f"  δ (tol):           {delta:.1e}")
    print("="*70 + "\n")

    results = run_frequency_sweep(
        levels_to_test=levels_to_test,
        N=N,
        V_obj=V_obj,
        f_min=f_min,
        f_max=f_max,
        n_freq=n_freq,
        zeta0=zeta0,
        mu1=mu1,
        max_iter=max_iter,
        delta=delta
    )

    print("\nAll frequency sweep computations done.")
    print("Figures and data saved in folder: freq_sweep_results/\n")
