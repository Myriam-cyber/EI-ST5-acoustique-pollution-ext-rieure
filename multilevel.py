# -*- coding: utf-8 -*-
"""
Acoustic Absorption Optimization for Multiple Fractal Levels
=============================================================

This script optimizes the placement of absorbing material on boundaries
of different fractal complexity levels (0=flat, 1=fractal, 2=higher fractal).

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

def project_Uad_star(chi_tentative, mask_R, beta_target, tol=1e-6, max_iter=50):
    """
    Projection P_{U*_ad(β)}(χ) = max(0, min(χ + ℓ, 1)) on Robin boundary,
    with ℓ chosen so that the mean on Γ (Robin) equals β = beta_target.
    
    This ensures: 
    - All values are in [0,1]
    - Volume constraint: mean(χ) = β
    """
    vals = chi_tentative[mask_R]
    S = vals.size
    if S == 0:
        return chi_tentative

    # Binary search for ℓ
    ell_min = -2.0
    ell_max = 2.0

    for iteration in range(max_iter):
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
    Final projection P_{U_ad(β)}(χ_relaxed) onto {0,1}:
    - Only modifies Robin boundary (mask_R)
    - Sets 1 on largest values to respect volume β
    - Sets 0 elsewhere
    
    beta_target = desired mean of χ on Γ_R
    """
    chi_bin = np.zeros_like(chi_relaxed)

    idx_i, idx_j = np.where(mask_R)
    vals = chi_relaxed[idx_i, idx_j]
    S = vals.size
    if S == 0:
        return chi_bin

    # Number of points to set to 1: β * S
    nb_ones = int(round(beta_target * S))
    nb_ones = max(0, min(nb_ones, S))

    # Indices of largest values
    order = np.argsort(vals)[::-1]  # descending
    sel = order[:nb_ones]

    chi_bin[idx_i[sel], idx_j[sel]] = 1.0

    return chi_bin


def compute_parametric_gradient(domain_omega, Alpha, u, p):
    """
    Parametric gradient g(x) = -Re(Alpha * p * u̅) on Γ_Robin.
    
    For interior approximation of boundary values, we average neighboring
    interior points.
    """
    (M, N) = np.shape(domain_omega)
    grad = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                vals_u = []
                vals_p = []
                
                # Neighbors (up, down, left, right)
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
                    # Gradient is -Re(Alpha * p * conj(u))
                    grad[i, j] = -np.real(Alpha * p_avg * np.conj(u_avg))
                else:
                    grad[i, j] = 0.0

    return grad


def compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    J(u,χ) = ∫_Ω |u|² dx + μ₁(Vol(χ) - V₀)²
    
    With penalty term to enforce volume constraint.
    """
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    E = np.sum(np.abs(u[mask_dom])**2) * (spacestep**2)

    if chi is not None:
        mask_R = (domain_omega == _env.NODE_ROBIN)
        Vol = np.sum(chi[mask_R])
        J = E + mu1 * (Vol - V_0)**2
    else:
        J = E

    return float(np.real(J))


def optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                          beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                          Alpha, zeta0, chi_init, V_obj, mu1, V_0,
                          max_iter=50, delta=1e-3, verbose=True):
    """
    Parametric gradient descent with adaptive step size.
    """
    (M, N) = np.shape(domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)

    numb_iter = max_iter
    energy = np.zeros((numb_iter+1, 1), dtype=np.float64)

    # Initialize
    chi = chi_init.copy()
    alpha_rob_curr = alpha_rob.copy()

    # Direct problem u(χ₀)
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob_curr)

    # Initial energy
    J = compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=chi)
    energy[0] = J
    
    if verbose:
        print(f"Initial energy J₀ = {J:.6e}")
        print(f"Initial β = {np.mean(chi[mask_R]):.4f} (target: {V_obj:.4f})")

    zeta = zeta0
    rel_tol = 1e-8

    for n in range(numb_iter):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {n}")
            print(f"{'='*60}")
            print(f"J = {J:.6e}, ζ = {zeta:.4e}, β = {np.mean(chi[mask_R]):.4f}")

        # 1) Adjoint problem p(χ_n)
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u,
                                     beta_pde, alpha_pde, alpha_dir,
                                     beta_neu, beta_rob, alpha_rob_curr)

        # 2) Gradient g_n(x) = -Re(Alpha * p * conj(u))|_Γ
        grad = compute_parametric_gradient(domain_omega, Alpha, u, p)
        
        grad_norm = np.linalg.norm(grad[mask_R])
        if verbose:
            print(f"||gradient||₂ = {grad_norm:.6e}")

        # 3) Gradient descent with adaptive step size
        best_J = J
        best_chi = chi
        best_u = u
        best_alpha_rob = alpha_rob_curr
        best_zeta = zeta
        found_better = False

        for j in range(10):
            zeta_trial = zeta / (2.0**j)

            chi_tent = chi + zeta_trial * grad
            chi_tent = project_Uad_star(chi_tent, mask_R, V_obj)
            alpha_rob_tent = Alpha * chi_tent

            u_tent = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                                f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir,
                                                beta_neu, beta_rob, alpha_rob_tent)
            
            J_tent = compute_objective_function(domain_omega, u_tent,
                                               spacestep, mu1, V_0, chi=chi_tent)

            if J_tent < best_J * (1.0 - rel_tol):
                best_J = J_tent
                best_chi = chi_tent
                best_u = u_tent
                best_alpha_rob = alpha_rob_tent
                best_zeta = zeta_trial
                found_better = True
                if verbose:
                    print(f"  Trial {j}: ζ = {zeta_trial:.4e}, J = {J_tent:.6e} ✓")
                break
            else:
                if verbose:
                    print(f"  Trial {j}: ζ = {zeta_trial:.4e}, J = {J_tent:.6e} ✗")

        if not found_better:
            if verbose:
                print("  No improvement found. Stopping.")
            energy[n+1:] = J
            break

        # Accept step
        chi_next = best_chi
        u_next = best_u
        alpha_rob_next = best_alpha_rob
        J_next = best_J

        # Adaptive step size
        zeta = min(best_zeta * 1.2, 2.0)

        energy[n+1] = J_next

        # Check convergence
        diff = np.max(np.abs(chi_next[mask_R] - chi[mask_R]))
        beta_current = np.mean(chi_next[mask_R])
        
        if verbose:
            print(f"  Accepted: J_new = {J_next:.6e} (ΔJ = {J - J_next:.6e})")
            print(f"  ||Δχ||_∞ = {diff:.6e}, β = {beta_current:.4f}")

        if diff < delta:
            if verbose:
                print(f"\n{'='*60}")
                print("CONVERGED: ||Δχ||_∞ < δ")
                print(f"{'='*60}")
            chi = chi_next
            u = u_next
            alpha_rob_curr = alpha_rob_next
            J = J_next
            energy[n+2:] = J
            break

        chi = chi_next
        u = u_next
        alpha_rob_curr = alpha_rob_next
        J = J_next

    if verbose:
        print(f"\nFinal energy J = {J:.6e}")
        print(f"Final β = {np.mean(chi[mask_R]):.4f}")
    
    return chi, energy, u, grad


def run_optimization_for_level(level, N, f_Hz, V_obj, zeta0, mu1, max_iter, delta):
    """
    Run complete optimization for a given fractal level.
    
    Parameters
    ----------
    level : int
        Fractal level (0=flat, 1=fractal, 2=higher fractal)
    N : int
        Grid resolution
    f_Hz : float
        Frequency in Hz
    V_obj : float
        Target volume fraction
    zeta0 : float
        Initial step size
    mu1 : float
        Penalty parameter
    max_iter : int
        Maximum iterations
    delta : float
        Convergence tolerance
        
    Returns
    -------
    results : dict
        Dictionary containing all results
    """
    print(f"\n{'#'*70}")
    print(f"# FRACTAL LEVEL {level}")
    print(f"{'#'*70}\n")
    
    start_time = time.time()
    
    # =================================================================
    # GEOMETRY
    # =================================================================
    M = 2 * N
    spacestep = 1.0 / N

    # =================================================================
    # PHYSICS
    # =================================================================
    c = 343.0  # Speed of sound [m/s]
    k_phys = 2*np.pi*f_Hz/c
    L_ref = 1.0  # Reference length [m]
    k = k_phys * L_ref
    omega = k
    print(f"Frequency = {f_Hz} Hz  →  k = {k:.3f} rad/unit")
    print(f"Grid: {M}×{N}, spacestep = {spacestep:.4f}")
    # =================================================================
    # MATERIAL PROPERTY: α(f) for concrete (BETON)
    # =================================================================
    freq_tab_alpha, alpha_tab_alpha = load_alpha_table(
        'dta_freq_MELAMINE.mtx',
        'dta_alpha_MELAMINE.mtx'
    )
    Alpha = alpha_of_freq(f_Hz, freq_tab_alpha, alpha_tab_alpha)
    print(f"Material α(f={f_Hz:.1f} Hz) = {Alpha.real:.4e} + {Alpha.imag:.4e}j")
    # =================================================================
    # PDE COEFFICIENTS
    # =================================================================
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # Count Robin boundary points
    mask_R = (domain_omega == _env.NODE_ROBIN)
    n_robin = np.sum(mask_R)
    print(f"Robin boundary points: {n_robin}")

    # =================================================================
    # SOURCE: Gaussian from top
    # =================================================================
# =================================================================
# SOURCE: localised Dirichlet on top boundary
# =================================================================
    # SOURCE: Autoroute en haut (modélisation bruit routier descendant)
    f[:, :] = 0.0
    f_neu[:, :] = 0.0
    f_rob[:, :] = 0.0
    f_dir[:, :] = 0.0

    i_source = 0
    n_sources = 6
    spacing = N // (n_sources + 1)

    amplitude = 2.0  # <-- augmenter ici (valeur typique : 10 à 50)
    for n in range(n_sources):
        j_src = (n + 1) * spacing
        if j_src < N:
            f_dir[i_source, j_src] = amplitude * (1.0 + 0j)

    for j in range(N):
        dist_center = abs(j - N // 2)
        f_dir[i_source, j] *= np.exp(-(dist_center / (N / 3)) ** 2)

    print(f"Placed {n_sources} sources (amplitude={amplitude}) on top boundary.")

    # =================================================================
    # INITIAL ROBIN CONDITION
    # =================================================================
    alpha_rob[:, :] = -omega * 1j

    # =================================================================
    # INITIAL DENSITY χ
    # =================================================================
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # =================================================================
    # ABSORBING MATERIAL COEFFICIENT
    # =================================================================
    #Alpha = 6.311111794566079 - 6.67835254158675*1j
    alpha_rob = Alpha * chi

    # =================================================================
    # OPTIMIZATION PARAMETERS
    # =================================================================
    S = np.sum(mask_R)
    V_0 = V_obj * S
    
    print(f"\nOptimization parameters:")
    print(f"  Target β = {V_obj:.2%}")
    print(f"  Initial ζ = {zeta0}")
    print(f"  Penalty μ₁ = {mu1:.2e}")
    print(f"  Max iterations = {max_iter}")
    print(f"  Convergence δ = {delta:.2e}")

    # =================================================================
    # UNCONTROLLED SOLUTION
    # =================================================================
    print(f"\n{'='*60}")
    print("SOLVING UNCONTROLLED PROBLEM")
    print(f"{'='*60}")
    
    u0 = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                    f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir,
                                    beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    
    J0 = compute_objective_function(domain_omega, u0, spacestep, mu1, V_0, chi=chi0)
    print(f"Uncontrolled energy J₀ = {J0:.6e}")

    # =================================================================
    # OPTIMIZATION
    # =================================================================
    print(f"\n{'='*60}")
    print("STARTING OPTIMIZATION")
    print(f"{'='*60}")
    
    chi_opt, energy, u_opt, grad = optimization_procedure(
        domain_omega, spacestep, omega,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, zeta0, chi, V_obj, mu1, V_0,
        max_iter=max_iter, delta=delta, verbose=True
    )

    chi_relaxed = chi_opt.copy()
    u_relaxed = u_opt.copy()

    # =================================================================
    # BINARY PROJECTION
    # =================================================================
    print(f"\n{'='*60}")
    print("BINARY PROJECTION")
    print(f"{'='*60}")
    
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
    print(f"Binary χ: β = {beta_bin:.4f}, J = {J_bin:.6e}")

    # =================================================================
    # SAVE PLOTS
    # =================================================================
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    # Create directory for this level
    output_dir = f'results_level_{level}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save uncontrolled solution
    postprocessing.myimshow(np.real(u0), 
                           title=f'Level {level}: Re(u₀)', 
                           colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, 
                           filename=f'{output_dir}/fig_u0_re.jpg')
    postprocessing.myimshow(chi0, 
                           title=f'Level {level}: χ₀', 
                           colorbar='colorbar', cmap='jet', vmin=0, vmax=1, 
                           filename=f'{output_dir}/fig_chi0.jpg')
    
    # Save controlled solution (binary)
    postprocessing.myimshow(np.real(u_bin), 
                           title=f'Level {level}: Re(u_opt)', 
                           colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, 
                           filename=f'{output_dir}/fig_un_re.jpg')
    postprocessing.myimshow(chi_bin, 
                           title=f'Level {level}: χ_opt (binary)', 
                           colorbar='colorbar', cmap='jet', vmin=0, vmax=1, 
                           filename=f'{output_dir}/fig_chin_binary.jpg')
    
    # Save relaxed solution
    postprocessing.myimshow(chi_relaxed, 
                           title=f'Level {level}: χ_opt (relaxed)', 
                           colorbar='colorbar', cmap='jet', vmin=0, vmax=1, 
                           filename=f'{output_dir}/fig_chin_relaxed.jpg')
    
    # Save error
    err = u_bin - u0
    postprocessing.myimshow(np.real(err), 
                           title=f'Level {level}: Re(u_opt - u₀)', 
                           colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, 
                           filename=f'{output_dir}/fig_err_real.jpg')
    
    # Save energy history
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
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print(f"LEVEL {level} OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Computation time:     {elapsed_time:.1f} seconds")
    print(f"Uncontrolled energy:  J₀ = {J0:.6e}")
    print(f"Relaxed energy:       J  = {J_relaxed:.6e}")
    print(f"Binary energy:        J  = {J_bin:.6e}")
    print(f"Improvement:          ΔJ/J₀ = {(J0 - J_bin)/J0:.2%}")
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
        'chi0': chi0,              # <<< AJOUT
        'u0': u0,
        'u_bin': u_bin,
        'domain': domain_omega,
        'computation_time': elapsed_time
    }



def compare_all_levels(results_list):
    """
    Create comparison plots for all levels.
    """
    print(f"\n{'#'*70}")
    print("# GENERATING COMPARISON PLOTS")
    print(f"{'#'*70}\n")
    
    output_dir = 'results_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    levels = [r['level'] for r in results_list]
    J0_values = [r['J0'] for r in results_list]
    J_bin_values = [r['J_bin'] for r in results_list]
    improvements = [r['improvement'] * 100 for r in results_list]
    n_robin_values = [r['n_robin'] for r in results_list]
    times = [r['computation_time'] for r in results_list]
    
    # 1. Energy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(levels))
    width = 0.35
    
    ax1.bar(x - width/2, J0_values, width, label='Uncontrolled (J₀)', alpha=0.8)
    ax1.bar(x + width/2, J_bin_values, width, label='Optimized (J_opt)', alpha=0.8)
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
    
    # 2. Complexity vs Performance
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
    
    # 3. Chi distribution comparison
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]
    
    for idx, result in enumerate(results_list):
        im = axes[idx].imshow(result['chi_bin'], cmap='jet', vmin=0, vmax=1)
        axes[idx].set_title(f"Level {result['level']}: χ_opt (binary)", fontsize=12)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chi_comparison.jpg', dpi=150)
    plt.close()
    
    # 4. Summary table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Level':<8} {'Robin pts':<12} {'J₀':<12} {'J_opt':<12} {'Improvement':<14} {'Time (s)':<10}")
    print(f"{'-'*70}")
    for r in results_list:
        print(f"{r['level']:<8} {r['n_robin']:<12} {r['J0']:<12.4e} {r['J_bin']:<12.4e} "
              f"{r['improvement']*100:<13.2f}% {r['computation_time']:<10.1f}")
    print(f"{'='*70}\n")
    
    # Save summary to file
    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write("ACOUSTIC OPTIMIZATION COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Level':<8} {'Robin pts':<12} {'J₀':<12} {'J_opt':<12} {'Improvement':<14} {'Time (s)':<10}\n")
        f.write("-"*70 + "\n")
        for r in results_list:
            f.write(f"{r['level']:<8} {r['n_robin']:<12} {r['J0']:<12.4e} {r['J_bin']:<12.4e} "
                   f"{r['improvement']*100:<13.2f}% {r['computation_time']:<10.1f}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"Comparison results saved to: {output_dir}/")


if __name__ == '__main__':
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  MULTI-LEVEL FRACTAL ACOUSTIC ABSORPTION OPTIMIZATION        ║
    ║  Comparison of plane and fractal boundary configurations      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # =================================================================
    # GLOBAL PARAMETERS (same for all levels)
    # =================================================================
    N = 50               # Grid resolution
    f_Hz = 180.0         # Frequency [Hz]
    V_obj = 0.4          # Target volume fraction (40%)
    zeta0 = 0.15          # Initial step size
    mu1 = 1e-9           # Volume penalty
    max_iter = 300     # Maximum iterations
    delta = 1e-4         # Convergence tolerance
    
    # Levels to test (0=flat, 1=fractal, 2=higher fractal)
    levels_to_test = [0, 1, 2, 3]
    
    print(f"\nGlobal Parameters:")
    print(f"  Grid resolution:   N = {N}")
    print(f"  Frequency:         f = {f_Hz} Hz")
    print(f"  Target volume:     β = {V_obj:.1%}")
    print(f"  Initial step size: ζ₀ = {zeta0}")
    print(f"  Penalty parameter: μ₁ = {mu1:.2e}")
    print(f"  Max iterations:    {max_iter}")
    print(f"  Convergence tol:   δ = {delta:.2e}")
    print(f"  Levels to test:    {levels_to_test}")
    
    # =================================================================
    # RUN OPTIMIZATION FOR ALL LEVELS
    # =================================================================
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
            print(f"\n{'!'*70}")
            print(f"ERROR in level {level}: {str(e)}")
            print(f"{'!'*70}\n")
            import traceback
            traceback.print_exc()
    
    # =================================================================
    # GENERATE COMPARISON PLOTS
    # =================================================================
    if len(results_all) > 0:
        compare_all_levels(results_all)
        
        print(f"\n{'='*70}")
        print("ALL OPTIMIZATIONS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nTotal levels computed: {len(results_all)}")
        print(f"Results saved in:")
        for r in results_all:
            print(f"  - results_level_{r['level']}/")
        print(f"  - results_comparison/")
        print("\n" + "="*70)
    else:
        print("\n{'!'*70}")
        print("NO RESULTS GENERATED - All optimizations failed")
        print("{'!'*70}\n")
    
    print('\nDone.')