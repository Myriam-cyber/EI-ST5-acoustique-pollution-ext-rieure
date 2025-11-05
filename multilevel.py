# -*- coding: utf-8 -*- 
"""
Acoustic Absorption Optimization for Multiple Fractal Levels
=============================================================

This script optimizes the placement of absorbing material on boundaries
of different fractal complexity levels (0=flat, 1=fractal, 2=higher fractal,...).

The optimization minimizes acoustic energy in the domain while respecting
a volume constraint on the absorbing material.
"""

# Python packages
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.io import mmread

# MRG packages
import _env
import preprocessing
import processing
import postprocessing


def adjust_beta_for_fractal_level(level, beta_flat, domain_omega, perimeter_level0=None):
    """
    Adjust the target beta based on the fractal level and boundary perimeter.
    If perimeter_level0 is not provided, it is calculated for level 0.
    """
    # Compte les points sur la frontière Robin
    mask_R = (domain_omega == _env.NODE_ROBIN)
    S = np.sum(mask_R)  # surface de la frontière Robin

    # Si perimeter_level0 n'a pas été passé, calculer le périmètre pour level 0 et le stocker
    if perimeter_level0 is None:
        perimeter_level0 = S  # Utilisation du périmètre pour le level 0 comme référence

    # Calculer le nombre de pixels absorbants à partir de beta_flat pour level 0
    total_pixels_level0 = int(beta_flat * perimeter_level0)

    # Pour les autres niveaux, ajuster beta pour obtenir le même nombre de pixels absorbants
    if level == 0:
        return beta_flat  # Pour le niveau plat, beta reste tel quel
    else:
        # Calculer le beta ajusté pour le niveau actuel afin de placer le même nombre de pixels
        beta_adjusted = total_pixels_level0 / S
        return beta_adjusted


# =====================================================================
#  ALPHA(f) : lecture et interpolation des données matériau
# =====================================================================
def load_alpha_table(freq_file, alpha_file):
    """Lit les fichiers .mtx contenant f (Hz) et alpha (complexe)."""
    freq_tab = np.array(mmread(freq_file)).flatten()
    alpha_tab = np.array(mmread(alpha_file)).flatten()

    idx = np.argsort(freq_tab)
    freq_tab = freq_tab[idx]
    alpha_tab = alpha_tab[idx]
    return freq_tab, alpha_tab


def alpha_of_freq(freq, freq_tab, alpha_tab):
    """Interpolation de α(f) complexe."""
    freq_clamped = np.clip(freq, freq_tab[0], freq_tab[-1])
    re = np.interp(freq_clamped, freq_tab, np.real(alpha_tab))
    im = np.interp(freq_clamped, freq_tab, np.imag(alpha_tab))
    return re + 1j * im


# =====================================================================
#  PROJECTIONS SUR LES ENSEMBLES ADMISSIBLES
# =====================================================================
def project_Uad_star(chi_tentative, mask_R, beta_target, tol=1e-6, max_iter=50):
    """
    Projection P_{U*_ad(β)}(χ) = max(0, min(χ + ℓ, 1)) sur la frontière Robin,
    avec ℓ choisi pour que la moyenne sur Γ_R soit β = beta_target.

    -> Garantit :
       - 0 <= chi <= 1
       - mean(chi|Γ_R) = β
    """
    vals = chi_tentative[mask_R]
    S = vals.size
    if S == 0:
        return chi_tentative

    # Dichotomie sur ℓ
    ell_min = -2.0
    ell_max = 2.0

    for _ in range(max_iter):
        ell_mid = 0.5 * (ell_min + ell_max)
        proj = np.clip(vals + ell_mid, 0.0, 1.0)
        m = proj.mean()
        if abs(m - beta_target) < tol:
            break
        if m > beta_target:
            ell_max = ell_mid
        else:
            ell_min = ell_mid

    proj_final = np.clip(vals + ell_mid, 0.0, 1.0)
    chi_new = chi_tentative.copy()
    chi_new[mask_R] = proj_final
    return chi_new


def project_to_binary(chi_relaxed, mask_R, beta_target):
    """
    Projection finale P_{U_ad(β)}(χ_relax) sur {0,1} :
    - On ne modifie que la frontière Robin (mask_R)
    - On met 1 sur les plus grandes valeurs, de sorte que
      la fraction de Robin couverte soit ≈ β
    """
    chi_bin = np.zeros_like(chi_relaxed)

    idx_i, idx_j = np.where(mask_R)
    vals = chi_relaxed[idx_i, idx_j]
    S = vals.size
    if S == 0:
        return chi_bin

    # Nombre de points "1" : β * S
    nb_ones = int(round(beta_target * S))
    nb_ones = max(0, min(nb_ones, S))

    order = np.argsort(vals)[::-1]  # tri décroissant
    sel = order[:nb_ones]

    chi_bin[idx_i[sel], idx_j[sel]] = 1.0
    return chi_bin


# =====================================================================
#  SOURCES : bruit d'autoroute en haut du domaine
# =====================================================================
def build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0, n_sources=6):
    """
    Modélisation du bruit d'autoroute sur le bord supérieur (Dirichlet).

    - n_sources 'voitures' réparties horizontalement,
    - pondération gaussienne pour densité plus forte au centre.
    """
    f[:, :] = 0.0
    f_neu[:, :] = 0.0
    f_rob[:, :] = 0.0
    f_dir[:, :] = 0.0

    M, N = f_dir.shape
    i_source = 0  # ligne du haut
    spacing = N // (n_sources + 1)

    # Sources ponctuelles
    for n in range(n_sources):
        j_src = (n + 1) * spacing
        if j_src < N:
            f_dir[i_source, j_src] = amplitude * (1.0 + 0.0j)

    # Gaussienne pour densité plus forte au centre
    for j in range(N):
        dist_center = abs(j - N // 2)
        f_dir[i_source, j] *= np.exp(-(dist_center / (N / 3)) ** 2)

    return f, f_dir, f_neu, f_rob


# =====================================================================
#  GRADIENT PARAMÉTRIQUE & FONCTIONNEL
# =====================================================================
def compute_parametric_gradient(domain_omega, Alpha, u, p):
    """
    Gradient paramétrique g(x) = -Re(Alpha * p * u̅) sur Γ_Robin.

    On approxime les valeurs "sur la frontière" par la moyenne des voisins
    intérieurs (4-voisinage).
    """
    (M, N) = np.shape(domain_omega)
    grad = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                vals_u = []
                vals_p = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < M and 0 <= jj < N:
                        if domain_omega[ii, jj] == _env.NODE_INTERIOR:
                            vals_u.append(u[ii, jj])
                            vals_p.append(p[ii, jj])

                if vals_u:
                    u_avg = sum(vals_u) / len(vals_u)
                    p_avg = sum(vals_p) / len(vals_p)
                    grad[i, j] = -np.real(Alpha * p_avg * np.conj(u_avg))
                else:
                    grad[i, j] = 0.0
    return grad


def compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    J(u,χ) = ∫_Ω |u|² dx + μ₁ (Vol(χ) - V₀)²

    avec Vol(χ) = somme de χ sur la frontière Robin.
    """
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    E = np.sum(np.abs(u[mask_dom]) ** 2) * (spacestep ** 2)

    if chi is not None:
        mask_R = (domain_omega == _env.NODE_ROBIN)
        Vol = np.sum(chi[mask_R])
        J = E + mu1 * (Vol - V_0) ** 2
    else:
        J = E

    return float(np.real(J))


# =====================================================================
#  DESCENTE DE GRADIENT AVEC ζ ADAPTATIF
# =====================================================================
def optimization_procedure(domain_omega, spacestep, omega,
                           f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir,
                           beta_neu, beta_rob, alpha_rob,
                           Alpha, zeta0, chi_init, V_obj, mu1, V_0,
                           max_iter=50, delta=1e-3, verbose=True):
    """
    Descente de gradient paramétrique avec ζ adaptatif :

      - backtracking sur J : on diminue ζ jusqu'à ce que J baisse,
      - puis ajustement de ζ en fonction de ||Δχ||_∞ :
          * si on a beaucoup bougé  -> ζ diminué
          * si on bouge très peu    -> ζ augmenté
          * sinon                   -> ζ conservé
    """
    (M, N) = np.shape(domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)

    numb_iter = max_iter
    energy = np.zeros((numb_iter + 1, 1), dtype=np.float64)

    chi = chi_init.copy()
    alpha_rob_curr = alpha_rob.copy()

    # Problème direct initial
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob_curr)

    J = compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=chi)
    energy[0] = J

    if verbose:
        print(f"Initial energy J0 = {J:.6e}")
        print(f"Initial β = {np.mean(chi[mask_R]):.4f} (target: {V_obj:.4f})")

    # bornes sur ζ
    zeta = zeta0
    zeta_min, zeta_max = 1e-5, 2.0

    for n in range(numb_iter):
        if verbose:
            print("\n" + "=" * 60)
            print(f"Iteration {n}")
            print("=" * 60)
            print(f"J = {J:.6e}, ζ = {zeta:.4e}, β = {np.mean(chi[mask_R]):.4f}")

        # 1) Problème adjoint
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u,
                                     beta_pde, alpha_pde, alpha_dir,
                                     beta_neu, beta_rob, alpha_rob_curr)

        # 2) Gradient paramétrique
        grad = compute_parametric_gradient(domain_omega, Alpha, u, p)
        grad_norm = np.linalg.norm(grad[mask_R])
        if verbose:
            print(f"||gradient||₂ = {grad_norm:.6e}")

        if grad_norm < 1e-14:
            if verbose:
                print("Gradient too small, stopping.")
            energy[n + 1:] = J
            break

        # 3) Backtracking sur J
        accepted = False
        zeta_trial = zeta

        for j in range(10):
            chi_tent = chi + zeta_trial * grad
            chi_tent = project_Uad_star(chi_tent, mask_R, V_obj)
            alpha_rob_tent = Alpha * chi_tent

            u_tent = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                                f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir,
                                                beta_neu, beta_rob, alpha_rob_tent)

            J_tent = compute_objective_function(domain_omega, u_tent,
                                                spacestep, mu1, V_0, chi=chi_tent)

            if verbose:
                print(f"  Trial {j}: ζ_trial = {zeta_trial:.4e}, J_trial = {J_tent:.6e}")

            if J_tent < J:
                best_J = J_tent
                best_chi = chi_tent
                best_u = u_tent
                best_alpha_rob = alpha_rob_tent
                accepted = True
                break
            else:
                zeta_trial *= 0.5

        if not accepted:
            if verbose:
                print("  No step found that decreases J. Stopping.")
            energy[n + 1:] = J
            break

        # 4) Variation de χ
        diff = np.max(np.abs(best_chi[mask_R] - chi[mask_R]))
        beta_current = np.mean(best_chi[mask_R])

        if verbose:
            print(f"  Accepted step: J_new = {best_J:.6e} (ΔJ = {J - best_J:.6e})")
            print(f"  ||Δχ||_∞ = {diff:.6e}, β = {beta_current:.4f}")

        # 5) Adaptation de ζ
        if diff > 0.1:
            zeta = max(zeta_trial * 0.5, zeta_min)
            if verbose:
                print(f"  Large move in χ (diff={diff:.3f}) → reducing ζ to {zeta:.4e}")
        elif diff < 0.05:
            zeta = min(zeta_trial * 1.5, zeta_max)
            if verbose:
                print(f"  Small move in χ (diff={diff:.3f}) → increasing ζ to {zeta:.4e}")
        else:
            zeta = zeta_trial
            if verbose:
                print(f"  Moderate move in χ (diff={diff:.3f}) → keeping ζ = {zeta:.4e}")

        # 6) Mise à jour
        chi = best_chi
        u = best_u
        alpha_rob_curr = best_alpha_rob
        J = best_J
        energy[n + 1] = J

        # 7) Critère d'arrêt
        if diff < delta:
            if verbose:
                print("\n" + "=" * 60)
                print("CONVERGED: ||Δχ||_∞ < δ")
                print("=" * 60)
            energy[n + 2:] = J
            break

    if verbose:
        print(f"\nFinal energy J = {J:.6e}")
        print(f"Final β = {np.mean(chi[mask_R]):.4f}")

    return chi, energy, u, grad



# =====================================================================
#  LANCEMENT POUR UN NIVEAU DE FRACTAL DONNÉ
# =====================================================================
def run_optimization_for_level(level, N, f_Hz, V_obj, zeta0, mu1, max_iter, delta):
    """
    Lance l'optimisation complète pour un niveau de fractal donné.
    """
    print("\n" + "#" * 70)
    print(f"# FRACTAL LEVEL {level}")
    print("#" * 70 + "\n")

    start_time = time.time()

    # Géométrie
    M = 2 * N
    spacestep = 1.0 / N

    # Physique
    c = 343.0
    k_phys = 2 * np.pi * f_Hz / c
    L_ref = 1.0
    k = k_phys * L_ref
    omega = k
    print(f"Frequency = {f_Hz} Hz  →  k = {k:.3f} rad/unit")
    print(f"Grid: {M}×{N}, spacestep = {spacestep:.4f}")

    # α(f)
    freq_tab_alpha, alpha_tab_alpha = load_alpha_table(
        'dta_freq_MELAMINE.mtx',
        'dta_alpha_MELAMINE.mtx'
    )
    Alpha = alpha_of_freq(f_Hz, freq_tab_alpha, alpha_tab_alpha)
    print(f"Material α(f={f_Hz:.1f} Hz) = {Alpha.real:.4e} + {Alpha.imag:.4e}j")

    # Coefficients PDE
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # Robin
    mask_R = (domain_omega == _env.NODE_ROBIN)
    n_robin = np.sum(mask_R)
    print(f"Robin boundary points: {n_robin}")

    # Calcul du volume cible en fonction du niveau fractal
    V_0 = adjust_beta_for_fractal_level(level, V_obj, domain_omega) * np.sum(mask_R)


    # Adaptation de ζ₀ selon le niveau de fractal
    if level == 0:
        zeta_level = zeta0 * 15.0
    elif level == 1:
        zeta_level = zeta0 * 1.0
    elif level == 2:
        zeta_level = zeta0 * 0.5
    else:  # level >= 3
        zeta_level = zeta0 * 0.15

    print(f"Using initial step size ζ₀(level={level}) = {zeta_level:.4e}")

    # Sources (autoroute en haut)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob,
                                               amplitude=2.0, n_sources=6)

    # Condition Robin initiale 'mur nu'
    alpha_rob[:, :] = -omega * 1j

    # Condition Robin "mur nu" (sera ensuite pondérée par χ)
    alpha_rob[:, :] = -omega * 1j

    # -----------------------------------------------------------------
    # DENSITÉ INITIALE χ₀
    # -----------------------------------------------------------------
    # 1) motif de départ (bande horizontale dans la fractale)
    chi_init = preprocessing._set_chi(M, N, x, y)
    chi_init = preprocessing.set2zero(chi_init, domain_omega)

    # 2) on impose le bon volume β = V_obj :
    #    - si tu veux du 0/1 dès le début -> project_to_binary
    #    - si tu préfères une densité continue -> project_Uad_star
    chi0 = project_to_binary(chi_init, mask_R, V_obj)
    # chi0 = project_Uad_star(chi_init, mask_R, V_obj)  # alternative continue

    beta0 = np.mean(chi0[mask_R])
    print(f"Initial chi0: beta0 = {beta0:.4f} (target {V_obj:.4f})")

    # 3) c'est χ₀ qui est utilisé pour le Robin matériel et pour J₀
    alpha_rob = Alpha * chi0


    print("\nOptimization parameters:")
    print(f"  Target β = {V_obj:.2%}")
    print(f"  Base ζ₀ = {zeta0}")
    print(f"  Effective ζ₀(level) = {zeta_level}")
    print(f"  Penalty μ₁ = {mu1:.2e}")
    print(f"  Max iterations = {max_iter}")
    print(f"  Convergence δ = {delta:.2e}")

    # Solution non contrôlée
    print("\n" + "=" * 60)
    print("SOLVING UNCONTROLLED PROBLEM")
    print("=" * 60)

    u0 = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                    f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir,
                                    beta_neu, beta_rob, alpha_rob)
    J0 = compute_objective_function(domain_omega, u0, spacestep, mu1, V_0, chi=chi0)
    print(f"Uncontrolled energy J0 = {J0:.6e}")

    # Optimisation
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION")
    print("=" * 60)

    chi_opt, energy, u_opt, grad = optimization_procedure(
        domain_omega, spacestep, omega,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, zeta_level, chi0, V_obj, mu1, V_0,
        max_iter=max_iter, delta=delta, verbose=True
    )

    chi_relaxed = chi_opt.copy()
    u_relaxed = u_opt.copy()

    # Projection binaire
    print("\n" + "=" * 60)
    print("BINARY PROJECTION")
    print("=" * 60)

    chi_bin = project_to_binary(chi_relaxed, mask_R, V_obj)
    alpha_rob_bin = Alpha * chi_bin

    u_bin = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                       f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, alpha_rob_bin)

    J_bin = compute_objective_function(domain_omega, u_bin, spacestep, mu1, V_0, chi=chi_bin)
    J_relaxed = compute_objective_function(domain_omega, u_relaxed, spacestep, mu1, V_0, chi=chi_relaxed)
    beta_bin = np.mean(chi_bin[mask_R])
    beta_relaxed = np.mean(chi_relaxed[mask_R])

    print(f"Relaxed χ: β = {beta_relaxed:.4f}, J = {J_relaxed:.6e}")
    print(f"Binary  χ: β = {beta_bin:.4f}, J = {J_bin:.6e}")

    # PLOTS
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    output_dir = f'results_level_{level}'
    os.makedirs(output_dir, exist_ok=True)

    # u0 et χ0
    postprocessing.myimshow(np.real(u0),
                            title=f'Level {level}: Re(u0)',
                            colorbar='colorbar', cmap='jet', vmin=-1, vmax=1,
                            filename=f'{output_dir}/fig_u0_re.jpg')
    postprocessing.myimshow(chi0,
                            title=f'Level {level}: chi0 (initial, beta={beta0:.2f})',
                            colorbar='colorbar', cmap='jet', vmin=0, vmax=1,
                            filename=f'{output_dir}/fig_chi0.jpg')

    # solution optimisée (binaire)
    postprocessing.myimshow(np.real(u_bin),
                            title=f'Level {level}: Re(u_opt)',
                            colorbar='colorbar', cmap='jet', vmin=-1, vmax=1,
                            filename=f'{output_dir}/fig_un_re.jpg')
    postprocessing.myimshow(chi_bin,
                            title=f'Level {level}: chi_opt (binary, beta={beta_bin:.2f})',
                            colorbar='colorbar', cmap='jet', vmin=0, vmax=1,
                            filename=f'{output_dir}/fig_chin_binary.jpg')

    # χ relaxée
    postprocessing.myimshow(chi_relaxed,
                            title=f'Level {level}: chi_opt (relaxed, beta={beta_relaxed:.2f})',
                            colorbar='colorbar', cmap='jet', vmin=0, vmax=1,
                            filename=f'{output_dir}/fig_chin_relaxed.jpg')

    # erreur
    err = u_bin - u0
    postprocessing.myimshow(np.real(err),
                            title=f'Level {level}: Re(u_opt - u0)',
                            colorbar='colorbar', cmap='jet', vmin=-1, vmax=1,
                            filename=f'{output_dir}/fig_err_real.jpg')

    # énergie vs itération
    plt.figure(figsize=(10, 6))
    plt.semilogy(energy, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy J', fontsize=12)
    plt.title(f'Level {level}: Energy History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_energy.jpg', dpi=150)
    plt.close()

    elapsed_time = time.time() - start_time

    # Résumé
    print("\n" + "=" * 60)
    print(f"LEVEL {level} OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Computation time:     {elapsed_time:.1f} seconds")
    print(f"Uncontrolled energy:  J0 = {J0:.6e}")
    print(f"Relaxed energy:       J  = {J_relaxed:.6e}")
    print(f"Binary energy:        J  = {J_bin:.6e}")
    print(f"Improvement:          ΔJ/J0 = {(J0 - J_bin)/J0:.2%}")
    print(f"Initial β (chi0):     {beta0:.4f}")
    print(f"Final β (relaxed):    {beta_relaxed:.4f}")
    print(f"Final β (binary):     {beta_bin:.4f}")
    print(f"Target β:             {V_obj:.4f}")
    print(f"Robin boundary pts:   {n_robin}")
    print(f"Results saved to:     {output_dir}/")

    return {
        'level': level,
        'n_robin': n_robin,
        'J0': J0,
        'J_relaxed': J_relaxed,
        'J_bin': J_bin,
        'improvement': (J0 - J_bin) / J0,
        'beta_relaxed': beta_relaxed,
        'beta_bin': beta_bin,
        'beta_target': V_obj,
        'energy_history': energy,
        'chi_bin': chi_bin,
        'chi_relaxed': chi_relaxed,
        'chi0': chi0,
        'u0': u0,
        'u_bin': u_bin,
        'domain': domain_omega,
        'computation_time': elapsed_time
    }



# =====================================================================
#  COMPARAISON ENTRE NIVEAUX
# =====================================================================
def compare_all_levels(results_list):
    """
    Crée des graphiques de comparaison pour tous les niveaux.
    """
    print("\n" + "#" * 70)
    print("# GENERATING COMPARISON PLOTS")
    print("#" * 70 + "\n")

    output_dir = 'results_comparison'
    os.makedirs(output_dir, exist_ok=True)

    levels = [r['level'] for r in results_list]
    J0_values = [r['J0'] for r in results_list]
    J_bin_values = [r['J_bin'] for r in results_list]
    improvements = [r['improvement'] * 100 for r in results_list]
    n_robin_values = [r['n_robin'] for r in results_list]
    times = [r['computation_time'] for r in results_list]

    # 1. Energie avant / après
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(levels))
    width = 0.35

    ax1.bar(x - width / 2, J0_values, width, label='Uncontrolled (J0)', alpha=0.8)
    ax1.bar(x + width / 2, J_bin_values, width, label='Optimized (J_opt)', alpha=0.8)
    ax1.set_xlabel('Fractal Level', fontsize=12)
    ax1.set_ylabel('Energy J', fontsize=12)
    ax1.set_title('Energy Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Level {l}' for l in levels])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(levels, improvements, alpha=0.8, color='green')
    ax2.set_xlabel('Fractal Level', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Energy Reduction', fontsize=14)
    ax2.set_xticks(levels)
    ax2.set_xticklabels([f'Level {l}' for l in levels])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_comparison.jpg', dpi=150)
    plt.close()

    # 2. Complexité / coût
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(levels, n_robin_values, 'o-', linewidth=2, markersize=10)
    ax1.set_xlabel('Fractal Level', fontsize=12)
    ax1.set_ylabel('Number of Robin Boundary Points', fontsize=12)
    ax1.set_title('Boundary Complexity', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(levels)

    ax2.plot(levels, times, 's-', linewidth=2, markersize=10, color='orange')
    ax2.set_xlabel('Fractal Level', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Computational Cost', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(levels)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_analysis.jpg', dpi=150)
    plt.close()

    # 3. Carte de χ_opt binaire pour chaque niveau
    fig, axes = plt.subplots(1, len(results_list), figsize=(6 * len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]

    for idx, result in enumerate(results_list):
        im = axes[idx].imshow(result['chi_bin'], cmap='jet', vmin=0, vmax=1)
        axes[idx].set_title(f"Level {result['level']}: chi_opt (binary)", fontsize=12)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/chi_comparison.jpg', dpi=150)
    plt.close()

    # 4. Tableau récapitulatif
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Level':<8} {'Robin pts':<12} {'J0':<12} {'J_opt':<12} "
          f"{'Improvement':<14} {'Time (s)':<10}")
    print("-" * 70)
    for r in results_list:
        print(f"{r['level']:<8} {r['n_robin']:<12} {r['J0']:<12.4e} {r['J_bin']:<12.4e} "
              f"{r['improvement']*100:<13.2f}% {r['computation_time']:<10.1f}")
    print("=" * 70 + "\n")

    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write("ACOUSTIC OPTIMIZATION COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Level':<8} {'Robin pts':<12} {'J0':<12} {'J_opt':<12} "
                f"{'Improvement':<14} {'Time (s)':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results_list:
            f.write(f"{r['level']:<8} {r['n_robin']:<12} {r['J0']:<12.4e} "
                    f"{r['J_bin']:<12.4e} {r['improvement']*100:<13.2f}% "
                    f"{r['computation_time']:<10.1f}\n")
        f.write("\n" + "=" * 70 + "\n")

    print(f"Comparison results saved to: {output_dir}/")


# =====================================================================
#  MAIN (optionnel)
# =====================================================================
if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  MULTI-LEVEL FRACTAL ACOUSTIC ABSORPTION OPTIMIZATION        ║
    ║  Comparison of plane and fractal boundary configurations      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    N = 50
    f_Hz = 180.0
    V_obj = 0.4
    zeta0 = 0.15
    mu1 = 1e-9
    max_iter = 300
    delta = 1e-4
    levels_to_test = [0, 1, 2, 3]

    print("\nGlobal Parameters:")
    print(f"  Grid resolution:   N = {N}")
    print(f"  Frequency:         f = {f_Hz} Hz")
    print(f"  Target volume:     β = {V_obj:.1%}")
    print(f"  Base step size:    ζ0 = {zeta0}")
    print(f"  Penalty parameter: μ1 = {mu1:.2e}")
    print(f"  Max iterations:    {max_iter}")
    print(f"  Convergence tol:   δ = {delta:.2e}")
    print(f"  Levels to test:    {levels_to_test}")

    results_all = []
    for level in levels_to_test:
        try:
            result = run_optimization_for_level(
                level=level,
                N=N,
                f_Hz=f_Hz,
                V_obj=V_obj,
                zeta0=zeta0,
                mu1=mu1,
                max_iter=max_iter,
                delta=delta
            )
            results_all.append(result)
        except Exception as e:
            print("\n" + "!" * 70)
            print(f"ERROR in level {level}: {str(e)}")
            print("!" * 70 + "\n")
            import traceback
            traceback.print_exc()

    if len(results_all) > 0:
        compare_all_levels(results_all)
        print("\n" + "=" * 70)
        print("ALL OPTIMIZATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nTotal levels computed: {len(results_all)}")
        print("Results saved in:")
        for r in results_all:
            print(f"  - results_level_{r['level']}/")
        print("  - results_comparison/")
        print("\n" + "=" * 70)
    else:
        print("\n" + "!" * 70)
        print("NO RESULTS GENERATED - All optimizations failed")
        print("!" * 70 + "\n")

    print('\nDone.')
