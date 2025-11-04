# -*- coding: utf-8 -*-
"""
Energy Analysis for Fully Absorbent Walls (BETON)
Frequency range automatically taken from dta_freq_BETON.mtx
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import mmread  # for reading .mtx files

# MRG packages
import _env
import preprocessing
import processing


# ============================================================
# Helper functions for alpha(f)
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
# Core computation
# ============================================================

def compute_energy_vs_frequency(domain_omega, spacestep, frequencies,
                                f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir,
                                beta_neu, beta_rob,
                                freq_tab_alpha, alpha_tab_alpha,
                                chi, wall_name="wall"):
    """Compute energy vs frequency for a fully absorbent wall."""
    c = 343.0
    L_ref = 1.0
    energies = np.zeros(len(frequencies), dtype=np.float64)

    print(f"\n{'='*60}")
    print(f"Computing energy for {wall_name} (fully absorbent, alpha=alpha(f))")
    print(f"Frequency range: [{frequencies[0]:.1f}, {frequencies[-1]:.1f}] Hz")
    print(f"Number of frequency points: {len(frequencies)}")
    print(f"{'='*60}\n")

    for idx, freq in enumerate(frequencies):
        k_phys = 2.0 * np.pi * freq / c
        omega = k_phys * L_ref

        Alpha_f = alpha_of_freq(freq, freq_tab_alpha, alpha_tab_alpha)
        alpha_rob = Alpha_f * chi

        u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                       f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, alpha_rob)

        mask_dom = (domain_omega == _env.NODE_INTERIOR)
        energy = np.sum(np.abs(u[mask_dom])**2) * (spacestep**2)
        energies[idx] = float(np.real(energy))

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(frequencies) - 1:
            print(f"  f = {freq:.1f} Hz → Energy = {energies[idx]:.6e}  "
                  f"(alpha = {Alpha_f.real:.3e} + {Alpha_f.imag:.3e}j)")

    return energies


def find_local_maxima(frequencies, energies, prominence_factor=0.05):
    """Detect local maxima in the energy spectrum."""
    maxima_indices = []
    for i in range(1, len(energies) - 1):
        if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]:
            min_neighbors = min(energies[i - 1], energies[i + 1])
            prominence = energies[i] - min_neighbors
            if prominence > prominence_factor * np.max(energies):
                maxima_indices.append(i)

    maxima_freqs = frequencies[maxima_indices]
    maxima_energies = energies[maxima_indices]
    return maxima_freqs, maxima_energies


def create_fractal_boundary(M, N, level, spacestep):
    """Create the domain geometry with flat or fractal Robin boundary."""
    domain_omega = np.zeros((M, N), dtype=np.int64)
    domain_omega[0:M, 0:N] = _env.NODE_INTERIOR
    domain_omega[0, 0:N] = _env.NODE_DIRICHLET  # top (incoming wave)
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
# MAIN SCRIPT
# ============================================================

if __name__ == '__main__':
    output_dir = 'energy_analysis_results'
    os.makedirs(output_dir, exist_ok=True)

    N = 50
    M = 2 * N
    spacestep = 1.0 / N

    # --- Load alpha(f) data for BETON
    freq_tab_alpha, alpha_tab_alpha = load_alpha_table(
        'dta_freq_BETON.mtx',
        'dta_alpha_BETON.mtx'
    )
    print("Loaded alpha(f) table for BETON:")
    print(f"  Frequency range: [{freq_tab_alpha[0]:.1f}, {freq_tab_alpha[-1]:.1f}] Hz")

    # --- Frequency range (auto or fixed)
    freq_min = max(100.0, freq_tab_alpha[0])
    freq_max = min(1000.0, freq_tab_alpha[-1])
    n_frequencies = 100
    frequencies = np.linspace(freq_min, freq_max, n_frequencies)

    # --- PDE setup
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob_base, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # --- Plane wave incidence (from top)
    f_dir[:, :] = 0.0
    theta_deg = 0.0
    theta = np.deg2rad(theta_deg)
    x_coords = np.arange(N) * spacestep
    k_ref = 2.0 * np.pi * 250.0 / 343.0
    phase_x = np.exp(1j * k_ref * np.sin(theta) * x_coords)
    f_dir[0, 0:N] = phase_x

    wall_shapes = [
        {'level': 0, 'name': 'Flat Wall'},
        {'level': 1, 'name': 'Fractal Level 1'},
        {'level': 2, 'name': 'Fractal Level 2'},
    ]

    all_results = {}

    # --- Compute for each wall shape
    for wall_config in wall_shapes:
        level = wall_config['level']
        wall_name = wall_config['name']

        print(f"\n{'#'*70}")
        print(f"# Processing: {wall_name}")
        print(f"{'#'*70}")

        domain_omega, x, y, shape_name = create_fractal_boundary(M, N, level, spacestep)
        chi = np.ones((M, N), dtype=np.float64)
        chi = preprocessing.set2zero(chi, domain_omega)

        energies = compute_energy_vs_frequency(
            domain_omega, spacestep, frequencies,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob,
            freq_tab_alpha, alpha_tab_alpha,
            chi, wall_name=wall_name
        )

        maxima_freqs, maxima_energies = find_local_maxima(frequencies, energies)

        all_results[shape_name] = {
            'frequencies': frequencies,
            'energies': energies,
            'maxima_freqs': maxima_freqs,
            'maxima_energies': maxima_energies,
            'wall_name': wall_name
        }

        # --- Plot for this shape ---
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, energies, linewidth=2.5, color='darkblue', label='Energy')
        if len(maxima_freqs) > 0:
            plt.plot(maxima_freqs, maxima_energies, 'ro', markersize=8, label='Local maxima')
            for freq, energy in zip(maxima_freqs, maxima_energies):
                plt.annotate(f'{freq:.0f} Hz', xy=(freq, energy),
                             xytext=(10, 10), textcoords='offset points',
                             fontsize=9, color='red',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='yellow', alpha=0.7))
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Acoustic Energy', fontsize=12)
        plt.title(f'Energy vs Frequency - {wall_name}\n(Fully Absorbent, α=α(f))',
                  fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        filename = f"energy_{shape_name.lower().replace(' ', '_')}_absorbent.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    # --- Global plot with all shapes ---
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green']
    for idx, (shape_name, results) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        plt.plot(results['frequencies'], results['energies'],
                 label=results['wall_name'], linewidth=2, color=color)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Acoustic Energy', fontsize=12)
    plt.title('Acoustic Energy vs Frequency - All Fractal Levels (Absorbent)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_all_shapes_absorbent.png'), dpi=300)
    plt.close()
    print("Saved: energy_all_shapes_absorbent.png")

    # --- Save summary ---
    with open(os.path.join(output_dir, 'energy_analysis_summary.txt'), 'w') as f_out:
        f_out.write("="*70 + "\n")
        f_out.write("ENERGY ANALYSIS SUMMARY FOR ABSORBENT WALLS (α = α(f), BETON)\n")
        f_out.write("="*70 + "\n\n")
        f_out.write(f"Frequency range: [{freq_min:.1f}, {freq_max:.1f}] Hz\n")
        f_out.write(f"Number of frequency points: {len(frequencies)}\n\n")

        for shape_name, res in all_results.items():
            f_out.write("-"*70 + "\n")
            f_out.write(f"Wall Shape: {res['wall_name']}\n")
            f_out.write("-"*70 + "\n")
            f_out.write(f"Mean energy: {np.mean(res['energies']):.6e}\n")
            f_out.write(f"Max energy:  {np.max(res['energies']):.6e}\n")
            f_out.write(f"Min energy:  {np.min(res['energies']):.6e}\n")
            f_out.write("Local Maxima:\n")
            if len(res['maxima_freqs']) > 0:
                for f_val, e_val in zip(res['maxima_freqs'], res['maxima_energies']):
                    f_out.write(f"  f = {f_val:.2f} Hz, E = {e_val:.6e}\n")
            else:
                f_out.write("  No significant local maxima found\n")
            f_out.write("\n")

    print(f"\n✅ Analysis complete! Results saved in '{output_dir}'")
