# -*- coding: utf-8 -*-
"""
Energy Analysis for Fully Absorbent Walls with Different Shapes
Frequency range: [100, 500] Hz
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Import required modules
import _env
import preprocessing
import processing
import postprocessing


def compute_energy_vs_frequency(domain_omega, spacestep, frequencies, 
                                  f, f_dir, f_neu, f_rob,
                                  beta_pde, alpha_pde, alpha_dir, 
                                  beta_neu, beta_rob, Alpha, 
                                  chi, wall_name="wall"):
    """
    Compute energy for a range of frequencies with fully absorbent wall (chi=1).
    
    Parameters:
    -----------
    domain_omega : array
        Domain geometry
    spacestep : float
        Spatial discretization step
    frequencies : array
        Array of frequencies to test (in Hz)
    chi : array
        Distribution of absorbent material (should be 1 for fully absorbent)
    wall_name : str
        Name of the wall shape for saving
        
    Returns:
    --------
    energies : array
        Energy values for each frequency
    """
    
    # Physical constants
    c = 343.0  # Speed of sound in m/s
    L_ref = 1.0
    
    # Storage for energies
    energies = np.zeros(len(frequencies))
    
    # Compute alpha_rob for fully absorbent wall
    alpha_rob = Alpha * chi
    
    print(f"\n{'='*60}")
    print(f"Computing energy for {wall_name}")
    print(f"Frequency range: [{frequencies[0]:.1f}, {frequencies[-1]:.1f}] Hz")
    print(f"Number of frequency points: {len(frequencies)}")
    print(f"{'='*60}\n")
    
    # Loop over frequencies
    for idx, freq in enumerate(frequencies):
        # Compute wavenumber
        k_phys = 2 * np.pi * freq / c
        omega = k_phys * L_ref
        
        # Solve Helmholtz equation
        u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                      f, f_dir, f_neu, f_rob,
                                      beta_pde, alpha_pde, alpha_dir,
                                      beta_neu, beta_rob, alpha_rob)
        
        # Compute energy
        mask_dom = (domain_omega == _env.NODE_INTERIOR)
        energy = np.sum(np.abs(u[mask_dom])**2) * (spacestep**2)
        energies[idx] = float(np.real(energy))
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(frequencies) - 1:
            print(f"  Frequency {idx+1}/{len(frequencies)}: {freq:.1f} Hz - Energy: {energies[idx]:.6e}")
    
    return energies


def find_local_maxima(frequencies, energies, prominence_factor=0.05):
    """
    Find local maxima in the energy curve.
    
    Parameters:
    -----------
    frequencies : array
        Frequency values
    energies : array
        Energy values
    prominence_factor : float
        Minimum prominence as fraction of max energy
        
    Returns:
    --------
    maxima_freqs : array
        Frequencies at local maxima
    maxima_energies : array
        Energy values at local maxima
    """
    maxima_indices = []
    
    # Find local maxima (simple peak detection)
    for i in range(1, len(energies) - 1):
        if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
            # Check prominence
            min_neighbors = min(energies[i-1], energies[i+1])
            prominence = energies[i] - min_neighbors
            if prominence > prominence_factor * np.max(energies):
                maxima_indices.append(i)
    
    maxima_freqs = frequencies[maxima_indices]
    maxima_energies = energies[maxima_indices]
    
    return maxima_freqs, maxima_energies


def create_fractal_boundary(M, N, level, spacestep):
    """
    Create different wall shapes including fractal boundaries.
    
    Parameters:
    -----------
    M, N : int
        Domain dimensions
    level : int
        Fractal generation level (0=flat, 1=first generation, etc.)
    spacestep : float
        Spatial step
        
    Returns:
    --------
    domain_omega : array
        Domain with specified boundary shape
    x, y : arrays
        Fractal coordinates
    shape_name : str
        Name of the shape
    """
    # Create base geometry
    domain_omega = np.zeros((M, N), dtype=np.int64)
    domain_omega[0:M, 0:N] = _env.NODE_INTERIOR
    domain_omega[0, 0:N] = _env.NODE_DIRICHLET  # north (top)
    domain_omega[M-1, 0:N] = _env.NODE_NEUMANN  # south (will be modified)
    domain_omega[0:M, 0] = _env.NODE_NEUMANN    # west
    domain_omega[0:M, N-1] = _env.NODE_NEUMANN  # east
    
    if level == 0:
        # Flat boundary
        domain_omega[N, 0:N] = _env.NODE_ROBIN
        x = np.arange(N)
        y = np.full(N, N)
        shape_name = "Flat"
    else:
        # Fractal boundary
        domain_omega[M-1, 0:N] = _env.NODE_NEUMANN
        nodes = preprocessing.create_fractal_nodes(
            [np.array([[0], [N]]), np.array([[N], [N]])], level
        )
        x, y = preprocessing.create_fractal_coordinates(nodes, domain_omega)
        
        # Define Robin boundary on fractal
        for k in range(0, len(x) - 1):
            domain_omega[int(y[k]), int(x[k])] = _env.NODE_ROBIN
        
        # Partition domain
        seed1 = [M-2, N-2]
        domain_omega = preprocessing.partition_domain(domain_omega, seed1)
        shape_name = f"Fractal_Level_{level}"
    
    return domain_omega, x, y, shape_name


if __name__ == '__main__':
    
    # Create output directory
    output_dir = 'energy_analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # === PARAMETERS ===
    N = 50
    M = 2 * N
    spacestep = 1.0 / N
    
    # Frequency range [100, 500] Hz
    freq_min = 100.0
    freq_max = 500.0
    n_frequencies = 41  # Number of frequency points
    frequencies = np.linspace(freq_min, freq_max, n_frequencies)
    
    # Physical constants
    c = 343.0
    
    # Material parameter (ISOREL-like)
    Alpha = 6.311111794566079 - 6.67835254158675*1j
    
    # === SETUP COEFFICIENTS ===
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob_base, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    
    # Setup source (plane wave at top)
    f_dir[:, :] = 0.0
    theta_deg = 0.0
    theta = np.deg2rad(theta_deg)
    x_coords = np.arange(N) * spacestep
    k_ref = 2 * np.pi * 250.0 / c  # Reference wavenumber for source
    phase_x = np.exp(1j * k_ref * np.sin(theta) * x_coords)
    f_dir[0, 0:N] = phase_x
    
    # === WALL SHAPES TO TEST ===
    wall_shapes = [
        {'level': 0, 'name': 'Flat Wall'},
        {'level': 1, 'name': 'Fractal Level 1'},
        {'level': 2, 'name': 'Fractal Level 2'},
    ]
    
    # Storage for all results
    all_results = {}
    
    # === COMPUTE ENERGY FOR EACH WALL SHAPE ===
    for wall_config in wall_shapes:
        level = wall_config['level']
        wall_name = wall_config['name']
        
        print(f"\n{'#'*70}")
        print(f"# Processing: {wall_name}")
        print(f"{'#'*70}")
        
        # Create domain with specified boundary shape
        domain_omega, x, y, shape_name = create_fractal_boundary(M, N, level, spacestep)
        
        # Create fully absorbent wall (chi = 1 everywhere on Robin boundary)
        chi = np.ones((M, N), dtype=np.float64)
        chi = preprocessing.set2zero(chi, domain_omega)
        
        # Compute energies
        energies = compute_energy_vs_frequency(
            domain_omega, spacestep, frequencies,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, Alpha,
            chi, wall_name=wall_name
        )
        
        # Find local maxima
        maxima_freqs, maxima_energies = find_local_maxima(frequencies, energies)
        
        # Store results
        all_results[shape_name] = {
            'frequencies': frequencies,
            'energies': energies,
            'maxima_freqs': maxima_freqs,
            'maxima_energies': maxima_energies,
            'wall_name': wall_name
        }
        
        # Print maxima
        print(f"\n  Local maxima for {wall_name}:")
        if len(maxima_freqs) > 0:
            for freq, energy in zip(maxima_freqs, maxima_energies):
                print(f"    f = {freq:.1f} Hz, E = {energy:.6e}")
        else:
            print("    No significant local maxima found")
    
    # === PLOTTING ===
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}\n")
    
    # Plot 1: All shapes on same plot
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (shape_name, results) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        plt.plot(results['frequencies'], results['energies'], 
                label=results['wall_name'], linewidth=2, color=color)
        
        # Mark local maxima
        if len(results['maxima_freqs']) > 0:
            plt.plot(results['maxima_freqs'], results['maxima_energies'], 
                    'o', markersize=8, color=color)
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Acoustic Energy', fontsize=12)
    plt.title('Acoustic Energy vs Frequency for Different Fully Absorbent Wall Shapes', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_all_shapes.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'energy_all_shapes.pdf'))
    print(f"Saved: energy_all_shapes.png and .pdf")
    plt.close()
    
    # Plot 2: Individual plots for each shape
    for shape_name, results in all_results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(results['frequencies'], results['energies'], 
                linewidth=2.5, color='darkblue')
        
        # Mark and annotate local maxima
        if len(results['maxima_freqs']) > 0:
            plt.plot(results['maxima_freqs'], results['maxima_energies'], 
                    'ro', markersize=10, label='Local Maxima')
            for freq, energy in zip(results['maxima_freqs'], results['maxima_energies']):
                plt.annotate(f'{freq:.0f} Hz', 
                           xy=(freq, energy),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Acoustic Energy', fontsize=12)
        plt.title(f'Acoustic Energy vs Frequency - {results["wall_name"]}\n(Fully Absorbent)', 
                 fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        if len(results['maxima_freqs']) > 0:
            plt.legend(fontsize=10)
        plt.tight_layout()
        
        filename = f"energy_{shape_name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        print(f"Saved: {filename}")
        plt.close()
    
    # Plot 3: Comparison of maxima
    plt.figure(figsize=(12, 7))
    bar_width = 0.25
    shapes_list = list(all_results.keys())
    
    # Create bar positions for each shape
    x_pos = np.arange(len(shapes_list))
    
    for idx, shape_name in enumerate(shapes_list):
        results = all_results[shape_name]
        if len(results['maxima_freqs']) > 0:
            # Plot bars for each maximum of this shape
            for max_idx, (freq, energy) in enumerate(zip(results['maxima_freqs'], 
                                                          results['maxima_energies'])):
                offset = max_idx * bar_width
                plt.bar(idx + offset, energy, bar_width, 
                       label=f"{results['wall_name']} - {freq:.0f} Hz" if max_idx < 3 else "")
    
    plt.xlabel('Wall Shape', fontsize=12)
    plt.ylabel('Energy at Local Maximum', fontsize=12)
    plt.title('Comparison of Local Maxima for Different Wall Shapes', 
             fontsize=14, fontweight='bold')
    plt.xticks(x_pos + bar_width, [all_results[s]['wall_name'] for s in shapes_list], 
              rotation=15, ha='right')
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'maxima_comparison.png'), dpi=300)
    print(f"Saved: maxima_comparison.png")
    plt.close()
    
    # === SAVE NUMERICAL RESULTS TO FILE ===
    with open(os.path.join(output_dir, 'energy_analysis_summary.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("ENERGY ANALYSIS SUMMARY FOR FULLY ABSORBENT WALLS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Frequency range: [{freq_min:.1f}, {freq_max:.1f}] Hz\n")
        f.write(f"Number of frequency points: {n_frequencies}\n")
        f.write(f"Material: ISOREL-like (Alpha = {Alpha})\n\n")
        
        for shape_name, results in all_results.items():
            f.write("-"*70 + "\n")
            f.write(f"Wall Shape: {results['wall_name']}\n")
            f.write("-"*70 + "\n")
            f.write(f"Mean energy: {np.mean(results['energies']):.6e}\n")
            f.write(f"Max energy: {np.max(results['energies']):.6e}\n")
            f.write(f"Min energy: {np.min(results['energies']):.6e}\n\n")
            
            f.write("Local Maxima:\n")
            if len(results['maxima_freqs']) > 0:
                for freq, energy in zip(results['maxima_freqs'], results['maxima_energies']):
                    f.write(f"  Frequency: {freq:.2f} Hz, Energy: {energy:.6e}\n")
            else:
                f.write("  No significant local maxima found\n")
            f.write("\n")
    
    print(f"\nSaved: energy_analysis_summary.txt")
    print(f"\n{'='*70}")
    print(f"Analysis complete! All results saved in '{output_dir}' directory")
    print(f"{'='*70}\n")