# -*- coding: utf-8 -*-
"""
Energy Analysis with Optimization for Frequency-Dependent Absorption
=====================================================================

This script performs energy vs frequency analysis comparing:
1. Fully absorbent wall (χ = 1 everywhere on Robin boundary)
2. Optimized relaxed solution (0 ≤ χ ≤ 1)
3. Optimized binary solution (χ ∈ {0,1})

For each frequency:
- Compute α(f) from material data
- Solve with fully absorbent wall
f- Optimize χ (relaxed solution)
- Project to binary solution
- Compute energy for all three cases
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.io import mmread

# MRG packages
import _env
import preprocessing
import processing


# ============================================================
# Alpha(f) Functions
# ============================================================

def load_alpha_table(freq_file, alpha_file):
    """Load frequency-dependent alpha values from .mtx files."""
    freq_tab = np.array(mmread(freq_file)).flatten()
    alpha_tab = np.array(mmread(alpha_file)).flatten()
    
    idx = np.argsort(freq_tab)
    freq_tab = freq_tab[idx]
    alpha_tab = alpha_tab[idx]
    return freq_tab, alpha_tab


def alpha_of_freq(freq, freq_tab, alpha_tab):
    """Interpolate complex alpha(f) for a given frequency."""
    freq_clamped = np.clip(freq, freq_tab[0], freq_tab[-1])
    re = np.interp(freq_clamped, freq_tab, np.real(alpha_tab))
    im = np.interp(freq_clamped, freq_tab, np.imag(alpha_tab))
    return re + 1j * im


# ============================================================
# Projection Functions
# ============================================================

def project_Uad_star(chi_tentative, mask_R, beta_target, tol=1e-6, max_iter=50):
    """
    Projection P_{U*_ad(β)}(χ) with volume constraint.
    Ensures: 0 ≤ χ ≤ 1 and mean(χ) = β on Robin boundary.
    """
    vals = chi_tentative[mask_R]
    S = vals.size
    if S == 0:
        return chi_tentative

    # Binary search for Lagrange multiplier ℓ
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
    Binary projection: set χ ∈ {0,1} respecting volume constraint.
    Selects largest values to achieve target mean β.
    """
    chi_bin = np.zeros_like(chi_relaxed)

    idx_i, idx_j = np.where(mask_R)
    vals = chi_relaxed[idx_i, idx_j]
    S = vals.size
    if S == 0:
        return chi_bin

    # Number of points to set to 1
    nb_ones = int(round(beta_target * S))
    nb_ones = max(0, min(nb_ones, S))

    # Select largest values
    order = np.argsort(vals)[::-1]
    sel = order[:nb_ones]

    chi_bin[idx_i[sel], idx_j[sel]] = 1.0

    return chi_bin


# ============================================================
# Energy and Gradient Computation
# ============================================================

def compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    J(u,χ) = ∫_Ω |u|² dx + μ₁(Vol(χ) - V₀)²
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


def compute_parametric_gradient(domain_omega, Alpha, u, p):
    """
    Parametric gradient g(x) = -Re(Alpha * p * conj(u)) on Γ_Robin.
    """
    (M, N) = np.shape(domain_omega)
    grad = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                vals_u = []
                vals_p = []
                
                # Check neighbors
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


# ============================================================
# Optimization Procedure
# ============================================================

def optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                          beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                          Alpha, zeta0, chi_init, V_obj, mu1, V_0,
                          max_iter=50, delta=1e-3, verbose=False):
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

    zeta = zeta0
    rel_tol = 1e-6

    for n in range(numb_iter):
        # 1) Adjoint problem
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u,
                                     beta_pde, alpha_pde, alpha_dir,
                                     beta_neu, beta_rob, alpha_rob_curr)

        # 2) Gradient
        grad = compute_parametric_gradient(domain_omega, Alpha, u, p)

        # 3) Gradient descent with line search
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
                break

        if not found_better:
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

        if diff < delta:
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
    
    return chi, energy, u


# ============================================================
# Geometry Creation
# ============================================================

def create_fractal_boundary(M, N, level, spacestep):
    """Create domain geometry with flat or fractal Robin boundary."""
    domain_omega = np.zeros((M, N), dtype=np.int64)
    domain_omega[0:M, 0:N] = _env.NODE_INTERIOR
    domain_omega[0, 0:N] = _env.NODE_DIRICHLET  # top
    domain_omega[M - 1, 0:N] = _env.NODE_NEUMANN  # bottom
    domain_omega[0:M, 0] = _env.NODE_NEUMANN
    domain_omega[0:M, N - 1] = _env.NODE_NEUMANN

    if level == 0:
        domain_omega[M - 1, 0:N] = _env.NODE_ROBIN
        x = np.arange(N)
        y = np.full(N, M - 1)
        shape_name = "Flat"
    else:
        nodes = preprocessing.create_fractal_nodes(
            [np.array([[0], [N]]), np.array([[N], [N]])], level
        )
        x, y = preprocessing.create_fractal_coordinates(nodes, domain_omega)
        for k in range(0, len(x) - 1):
            domain_omega[int(y[k]), int(x[k])] = _env.NODE_ROBIN
        domain_omega = preprocessing.partition_domain(domain_omega, [M - 2, N - 2])
        shape_name = f"Fractal_Level_{level}"
    
    return domain_omega, x, y, shape_name


# ============================================================
# MAIN COMPUTATION
# ============================================================

def compute_energy_all_cases(domain_omega, spacestep, frequencies,
                             f, f_dir, f_neu, f_rob,
                             beta_pde, alpha_pde, alpha_dir,
                             beta_neu, beta_rob,
                             freq_tab_alpha, alpha_tab_alpha,
                             V_obj, zeta0, mu1, max_iter, delta,
                             wall_name="wall", verbose=True):
    """
    Compute energy vs frequency for THREE cases:
    1. Fully absorbent wall (χ = 1 everywhere)
    2. Optimized relaxed solution
    3. Optimized binary solution
    """
    c = 343.0
    L_ref = 1.0
    
    energies_absorbent = np.zeros(len(frequencies), dtype=np.float64)
    energies_relaxed = np.zeros(len(frequencies), dtype=np.float64)
    energies_binary = np.zeros(len(frequencies), dtype=np.float64)
    
    mask_R = (domain_omega == _env.NODE_ROBIN)
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    S = np.sum(mask_R)
    V_0 = V_obj * S
    
    (M, N) = np.shape(domain_omega)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Computing energy for {wall_name}")
        print(f"Frequency range: [{frequencies[0]:.1f}, {frequencies[-1]:.1f}] Hz")
        print(f"Number of frequency points: {len(frequencies)}")
        print(f"Target volume fraction for optimization: β = {V_obj:.2%}")
        print(f"{'='*70}\n")

    for idx, freq in enumerate(frequencies):
        if verbose:
            print(f"\n--- Frequency {idx+1}/{len(frequencies)}: f = {freq:.1f} Hz ---")
        
        # 1. Compute α(f) for this frequency
        Alpha_f = alpha_of_freq(freq, freq_tab_alpha, alpha_tab_alpha)
        
        # 2. Compute k and omega
        k_phys = 2.0 * np.pi * freq / c
        omega = k_phys * L_ref
        
        # ========================================================
        # CASE 1: FULLY ABSORBENT WALL (χ = 1 everywhere)
        # ========================================================
        chi_absorbent = np.ones((M, N), dtype=np.float64)
        chi_absorbent = preprocessing.set2zero(chi_absorbent, domain_omega)
        alpha_rob_absorbent = Alpha_f * chi_absorbent
        
        u_absorbent = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, alpha_rob_absorbent
        )
        
        E_absorbent = np.sum(np.abs(u_absorbent[mask_dom])**2) * (spacestep**2)
        energies_absorbent[idx] = float(np.real(E_absorbent))
        
        # ========================================================
        # CASE 2 & 3: OPTIMIZED SOLUTIONS
        # ========================================================
        # Initialize χ for optimization (uniform distribution)
        chi_init = np.ones((M, N), dtype=np.float64) * V_obj
        chi_init = preprocessing.set2zero(chi_init, domain_omega)
        alpha_rob = Alpha_f * chi_init
        
        # Optimize
        chi_relaxed, energy_hist, u_relaxed = optimization_procedure(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
            Alpha_f, zeta0, chi_init, V_obj, mu1, V_0,
            max_iter=max_iter, delta=delta, verbose=False
        )
        
        # Compute energy for RELAXED solution
        E_relaxed = np.sum(np.abs(u_relaxed[mask_dom])**2) * (spacestep**2)
        energies_relaxed[idx] = float(np.real(E_relaxed))
        
        # Binary projection
        chi_binary = project_to_binary(chi_relaxed, mask_R, V_obj)
        alpha_rob_binary = Alpha_f * chi_binary
        
        # Solve with BINARY χ
        u_binary = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, alpha_rob_binary
        )
        
        # Compute energy for BINARY solution
        E_binary = np.sum(np.abs(u_binary[mask_dom])**2) * (spacestep**2)
        energies_binary[idx] = float(np.real(E_binary))
        
        if verbose and ((idx + 1) % 5 == 0 or idx == 0 or idx == len(frequencies) - 1):
            beta_rel = np.mean(chi_relaxed[mask_R])
            beta_bin = np.mean(chi_binary[mask_R])
            print(f"  α(f) = {Alpha_f.real:.3e} + {Alpha_f.imag:.3e}j")
            print(f"  E_absorbent = {E_absorbent:.6e} (χ=1)")
            print(f"  E_relaxed   = {E_relaxed:.6e} (β={beta_rel:.4f})")
            print(f"  E_binary    = {E_binary:.6e} (β={beta_bin:.4f})")

    return energies_absorbent, energies_relaxed, energies_binary


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
# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == '__main__':
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  ENERGY vs FREQUENCY ANALYSIS: COMPARISON OF THREE CASES         ║
    ║  1. Fully Absorbent Wall (χ=1)                                   ║
    ║  2. Optimized Relaxed Solution (0≤χ≤1)                           ║
    ║  3. Optimized Binary Solution (χ∈{0,1})                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    output_dir = 'energy_optimization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # PARAMETERS
    # ============================================================
    N = 50
    M = 2 * N
    spacestep = 1.0 / N
    
    # Load alpha(f) data
    freq_tab_alpha, alpha_tab_alpha = load_alpha_table(
        'dta_freq_MELAMINE.mtx',
        'dta_alpha_MELAMINE.mtx'
    )
    print(f"\nLoaded α(f) data:")
    print(f"  Frequency range: [{freq_tab_alpha[0]:.1f}, {freq_tab_alpha[-1]:.1f}] Hz")
    
    # Frequency range for analysis
    freq_min = max(100.0, freq_tab_alpha[0])
    freq_max = min(700.0, freq_tab_alpha[-1])
    n_frequencies = 200  # Reduced for faster computation
    frequencies = np.linspace(freq_min, freq_max, n_frequencies)
    
    print(f"\nAnalysis parameters:")
    print(f"  Grid: {M}×{N}, spacestep = {spacestep:.4f}")
    print(f"  Frequency range: [{freq_min:.1f}, {freq_max:.1f}] Hz")
    print(f"  Number of frequencies: {n_frequencies}")
    
    # Optimization parameters
    V_obj = 0.4      # Target volume fraction
    zeta0 = 0.7      # Initial step size
    mu1 = 1e-10      # Penalty parameter
    max_iter = 100   # Max iterations per frequency
    delta = 5e-5     # Convergence tolerance
    
    print(f"\nOptimization parameters:")
    print(f"  Target β = {V_obj:.1%}")
    print(f"  Initial ζ = {zeta0}")
    print(f"  Penalty μ₁ = {mu1:.2e}")
    print(f"  Max iterations = {max_iter}")
    print(f"  Convergence δ = {delta:.2e}")
    
    # ============================================================
    # PDE SETUP
    # ============================================================
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob_base, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    
    #source
    # =====================================================================
    #  SOURCES : bruit d'autoroute en haut du domaine
    # ===============================

    # Sources (autoroute en haut)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob,
                                                amplitude=2.0, n_sources=6)

    
    # ============================================================
    # WALL CONFIGURATIONS
    # ============================================================
    wall_shapes = [
        {'level': 0, 'name': 'Flat Wall'},
        {'level': 1, 'name': 'Fractal Level 1'},
        {'level': 2, 'name': 'Fractal Level 2'},
    ]
    
    all_results = {}
    
    # ============================================================
    # COMPUTE FOR EACH WALL SHAPE
    # ============================================================
    for wall_config in wall_shapes:
        level = wall_config['level']
        wall_name = wall_config['name']
        
        print(f"\n{'#'*70}")
        print(f"# Processing: {wall_name}")
        print(f"{'#'*70}")
        
        start_time = time.time()
        
        # Create geometry
        domain_omega, x, y, shape_name = create_fractal_boundary(M, N, level, spacestep)
        
        # Compute all three cases
        energies_absorbent, energies_relaxed, energies_binary = compute_energy_all_cases(
            domain_omega, spacestep, frequencies,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob,
            freq_tab_alpha, alpha_tab_alpha,
            V_obj, zeta0, mu1, max_iter, delta,
            wall_name=wall_name, verbose=True
        )
        
        elapsed = time.time() - start_time
        
        all_results[shape_name] = {
            'frequencies': frequencies,
            'energies_absorbent': energies_absorbent,
            'energies_relaxed': energies_relaxed,
            'energies_binary': energies_binary,
            'wall_name': wall_name,
            'computation_time': elapsed
        }
        
        print(f"\n✓ {wall_name} completed in {elapsed:.1f} seconds")
        print(f"  Mean E_absorbent: {np.mean(energies_absorbent):.6e}")
        print(f"  Mean E_relaxed:   {np.mean(energies_relaxed):.6e}")
        print(f"  Mean E_binary:    {np.mean(energies_binary):.6e}")
        
        # ============================================================
        # PLOT FOR THIS SHAPE (3 curves)
        # ============================================================
        plt.figure(figsize=(14, 8))
        
        plt.plot(frequencies, energies_absorbent, linewidth=2.5, 
                color='black', label='Fully Absorbent (χ=1)', alpha=0.8, linestyle='-')
        plt.plot(frequencies, energies_relaxed, linewidth=2.5, 
                color='blue', label=f'Optimized Relaxed (β={V_obj:.1%})', alpha=0.8, linestyle='-')
        plt.plot(frequencies, energies_binary, linewidth=2.5, 
                color='red', label=f'Optimized Binary (β={V_obj:.1%})', alpha=0.8, linestyle='--')
        
        plt.xlabel('Frequency (Hz)', fontsize=13)
        plt.ylabel('Acoustic Energy', fontsize=13)
        plt.title(f'Energy vs Frequency - {wall_name}\n' + 
                 f'Comparison: Fully Absorbent vs Optimized Solutions',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        filename = f"energy_{shape_name.lower().replace(' ', '_')}_comparison.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"  Saved: {filename}")
    
    # ============================================================
    # GLOBAL COMPARISON PLOTS
    # ============================================================
    
    # Plot 1: All absorbent solutions
    plt.figure(figsize=(14, 7))
    colors_abs = ['black', 'gray', 'darkgray']
    for idx, (shape_name, results) in enumerate(all_results.items()):
        color = colors_abs[idx % len(colors_abs)]
        plt.plot(results['frequencies'], results['energies_absorbent'],
                 label=f"{results['wall_name']} (Absorbent)", 
                 linewidth=2, color=color, alpha=0.7)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Acoustic Energy', fontsize=12)
    plt.title('Energy vs Frequency - All Levels (Fully Absorbent χ=1)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_all_absorbent.png'), dpi=300)
    plt.close()
    print("\nSaved: energy_all_absorbent.png")
    
    # Plot 2: All relaxed solutions
    plt.figure(figsize=(14, 7))
    colors_rel = ['blue', 'green', 'purple']
    for idx, (shape_name, results) in enumerate(all_results.items()):
        color = colors_rel[idx % len(colors_rel)]
        plt.plot(results['frequencies'], results['energies_relaxed'],
                 label=f"{results['wall_name']} (Relaxed)", 
                 linewidth=2, color=color, alpha=0.8)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Acoustic Energy', fontsize=12)
    plt.title(f'Energy vs Frequency - All Levels (Relaxed χ, β={V_obj:.1%})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_all_relaxed.png'), dpi=300)
    plt.close()
    print("Saved: energy_all_relaxed.png")
    
    # Plot 3: All binary solutions
    plt.figure(figsize=(14, 7))
    colors_bin = ['red', 'orange', 'brown']
    for idx, (shape_name, results) in enumerate(all_results.items()):
        color = colors_bin[idx % len(colors_bin)]
        plt.plot(results['frequencies'], results['energies_binary'],
                 label=f"{results['wall_name']} (Binary)", 
                 linewidth=2, color=color, alpha=0.8, linestyle='--')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Acoustic Energy', fontsize=12)
    plt.title(f'Energy vs Frequency - All Levels (Binary χ, β={V_obj:.1%})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_all_binary.png'), dpi=300)
    plt.close()
    print("Saved: energy_all_binary.png")
    
    # Plot 4: Combined comparison (one level at a time, all three curves)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (shape_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.plot(results['frequencies'], results['energies_absorbent'],
               linewidth=2.5, color='black', label='Absorbent (χ=1)', alpha=0.8)
        ax.plot(results['frequencies'], results['energies_relaxed'],
               linewidth=2.5, color='blue', label=f'Relaxed (β={V_obj:.1%})', alpha=0.8)
        ax.plot(results['frequencies'], results['energies_binary'],
               linewidth=2.5, color='red', label=f'Binary (β={V_obj:.1%})', alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Acoustic Energy', fontsize=11)
        ax.set_title(results['wall_name'], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_comparison_all_levels.png'), dpi=300)
    plt.close()
    print("Saved: energy_comparison_all_levels.png")