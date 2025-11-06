
# Python packages
import matplotlib.pyplot as plt
import numpy
import os
# MRG packages
import _env
import preprocessing
import processing
import postprocessing

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
    ell_min = -2.0  # Increased range for robustness
    ell_max = 2.0

    for iteration in range(max_iter):
        ell_mid = 0.5 * (ell_min + ell_max)
        proj = numpy.clip(vals + ell_mid, 0.0, 1.0)
        m = proj.mean()
        
        if abs(m - beta_target) < tol:
            break
            
        if m > beta_target:
            ell_max = ell_mid
        else:
            ell_min = ell_mid

    proj_final = numpy.clip(vals + ell_mid, 0.0, 1.0)
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
    # Note: Using 'np' as imported in your main script
    chi_bin = numpy.zeros_like(chi_relaxed)

    # Get coordinates and "fuzzy" values from the boundary
    idx_i, idx_j = numpy.where(mask_R)
    vals_relaxed = chi_relaxed[idx_i, idx_j]
    
    S = vals_relaxed.size
    if S == 0:
        return chi_bin

    # --- Original Binary Projection Logic ---
    nb_ones = int(round(beta_target * S))
    nb_ones = max(0, min(nb_ones, S))
    order = numpy.argsort(vals_relaxed)[::-1]  # descending
    sel = order[:nb_ones]
    chi_bin[idx_i[sel], idx_j[sel]] = 1.0
    
    # --- START: New Plotting Code ---
    
    # 1. Get the new binary values from the same boundary points
    vals_bin = chi_bin[idx_i, idx_j]

    # 2. Sort all data by position (j-index) for a clean 1D plot
    # This ensures the x-axis represents the position along the boundary
    sort_indices = numpy.argsort(idx_j)
    
    x_position = idx_j[sort_indices]
    y_relaxed = vals_relaxed[sort_indices]
    y_binary = vals_bin[sort_indices]

    # 3. Create the plot
    plt.figure(figsize=(15, 7))
    plt.title(f"Binary Projection (Target β = {beta_target:.2%})")
    
    # Plot the "fuzzy" relaxed values
    plt.plot(x_position, y_relaxed, 'b-', label='Relaxed χ (Input)', alpha=0.7)
    
    # Plot the final binary values as 'step' or 'markers' for clarity
    # Using 'ro' (red 'o' markers) shows the discrete 0/1 nature
    plt.plot(x_position, y_binary, 'ro', label='Binary χ (Output)', markersize=3)
    
    plt.xlabel("Position on Boundary (j-index)")
    plt.ylabel("χ Value (Material Density)")
    plt.ylim(-0.1, 1.1) # Give padding to 0 and 1
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # 4. Show the plot during execution
    plt.show()

    # --- END: New Plotting Code ---

    return chi_bin


def compute_parametric_gradient(domain_omega, Alpha, u, p):
    """
    Parametric gradient g(x) = -Re(Alpha * p * u̅) on Γ_Robin.
    
    This is derived from the Lagrangian method (equation 7.42):
    ⟨J'(χ), χ₀⟩ = -∫_Γ χ₀ Re(α u(χ) p(χ)) dμ
    
    For interior approximation of boundary values, we average neighboring
    interior points.
    """
    (M, N) = numpy.shape(domain_omega)
    grad = numpy.zeros((M, N), dtype=numpy.float64)

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
                    grad[i, j] = -numpy.real(Alpha * p_avg * numpy.conj(u_avg))
                else:
                    grad[i, j] = 0.0

    return grad


def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    J(u,χ) = ∫_Ω |u|² dx + μ₁(Vol(χ) - V₀)
    
    With penalty term to enforce volume constraint.
    """
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    E = numpy.sum(numpy.abs(u[mask_dom])**2) * (spacestep**2)

    if chi is not None:
        mask_R = (domain_omega == _env.NODE_ROBIN)
        Vol = numpy.sum(chi[mask_R])
        J = E + mu1 * (Vol - V_0)**2  # Quadratic penalty for volume
    else:
        J = E

    return float(numpy.real(J))


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, zeta0, chi_init, V_obj, mu1, V_0,
                           max_iter=50, delta=1e-3):
    """
    Parametric gradient descent with adaptive step size.
    
    PARAMETERS TO TUNE:
    -------------------
    max_iter : int (default=50)
        Maximum number of iterations
    
    delta : float (default=1e-3)
        Convergence criterion: max|χ_{n+1} - χ_n| < delta
    
    zeta0 : float (default=0.5)
        Initial step size
    
    mu1 : float (default=1e-5)
        Penalty parameter for volume constraint
    
    ADAPTIVE STEP SIZE STRATEGY:
    ----------------------------
    - If energy decreases: increase step size slightly
    - If energy increases: reduce step size by half
    - Maximum 10 halvings per iteration
    
    ALGORITHM:
    ----------
    1. Solve direct problem: u(χ_n)
    2. Solve adjoint problem: p(χ_n)
    3. Compute gradient: g_n = -Re(α p u̅)
    4. Update: χ_tent = χ_n + ζ_n g_n
    5. Project: χ_{n+1} = P_{U*_ad(β)}(χ_tent)
    6. Adaptive step size adjustment
    """

    (M, N) = numpy.shape(domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)

    numb_iter = max_iter
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)

    # Initialize
    chi = chi_init.copy()
    alpha_rob_curr = alpha_rob.copy()

    # Direct problem u(χ₀)
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob_curr)

    # Initial energy
    J = your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=chi)
    energy[0] = J
    
    print(f"Initial energy J₀ = {J:.6e}")
    print(f"Initial β = {numpy.mean(chi[mask_R]):.4f} (target: {V_obj:.4f})")

    zeta = zeta0
    rel_tol = 1e-6  # Relative tolerance for energy improvement

    for n in range(numb_iter):
        print(f"\n{'='*60}")
        print(f"Iteration {n}")
        print(f"{'='*60}")
        print(f"J = {J:.6e}, ζ = {zeta:.4e}, β = {numpy.mean(chi[mask_R]):.4f}")

        # 1) Adjoint problem p(χ_n)
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u,
                                     beta_pde, alpha_pde, alpha_dir,
                                     beta_neu, beta_rob, alpha_rob_curr)

        # 2) Gradient g_n(x) = -Re(Alpha * p * conj(u))|_Γ
        grad = compute_parametric_gradient(domain_omega, Alpha, u, p)
        
        grad_norm = numpy.linalg.norm(grad[mask_R])
        print(f"||gradient||₂ = {grad_norm:.6e}")

        # 3) Gradient descent with adaptive step size
        best_J = J
        best_chi = chi
        best_u = u
        best_alpha_rob = alpha_rob_curr
        best_zeta = zeta
        found_better = False

        # Try current step size and reduce if needed
        for j in range(10):  # Max 10 halvings
            zeta_trial = zeta / (2.0**j)

            # Update: χ_tent = χ_n + ζ * grad
            chi_tent = chi + zeta_trial * grad
            
            # Project to maintain constraints
            chi_tent = project_Uad_star(chi_tent, mask_R, V_obj)
            alpha_rob_tent = Alpha * chi_tent

            # Solve direct problem with updated χ
            u_tent = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                                f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir,
                                                beta_neu, beta_rob, alpha_rob_tent)
            
            J_tent = your_compute_objective_function(domain_omega, u_tent,
                                                     spacestep, mu1, V_0, chi=chi_tent)

            # Accept if energy decreased
            if J_tent < best_J * (1.0 - rel_tol):
                best_J = J_tent
                best_chi = chi_tent
                best_u = u_tent
                best_alpha_rob = alpha_rob_tent
                best_zeta = zeta_trial
                found_better = True
                print(f"  Trial {j}: ζ = {zeta_trial:.4e}, J = {J_tent:.6e} ✓")
                break
            else:
                print(f"  Trial {j}: ζ = {zeta_trial:.4e}, J = {J_tent:.6e} ✗")

        if not found_better:
            print("  No improvement found. Stopping.")
            energy[n+1:] = J
            break

        # Accept step
        chi_next = best_chi
        u_next = best_u
        alpha_rob_next = best_alpha_rob
        J_next = best_J

        # Adaptive step size: increase if successful
        zeta = min(best_zeta * 1.2, 2.0)  # Increase by 20%, cap at 2.0

        energy[n+1] = J_next

        # Check convergence
        diff = numpy.max(numpy.abs(chi_next[mask_R] - chi[mask_R]))
        beta_current = numpy.mean(chi_next[mask_R])
        
        print(f"  Accepted: J_new = {J_next:.6e} (ΔJ = {J - J_next:.6e})")
        print(f"  ||Δχ||_∞ = {diff:.6e}, β = {beta_current:.4f}")

        if diff < delta:
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

    print(f"\nFinal energy J = {J:.6e}")
    print(f"Final β = {numpy.mean(chi[mask_R]):.4f}")
    print('Computing final solution u...')
    
    return chi, energy, u, grad

# source
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
        f_dir[i_source, j] *= numpy.exp(-(dist_center / (N / 3)) ** 2)

    return f, f_dir, f_neu, f_rob


if __name__ == '__main__':

    # =================================================================
    # GEOMETRY
    # =================================================================
    N = 50  # Grid points in x-direction
    M = 2 * N  # Grid points in y-direction (2:1 aspect ratio)
    level = 0  # Fractal level (0 = flat boundary)
    spacestep = 1.0 / N

    # =================================================================
    # PHYSICS: 125 Hz, c = 343 m/s
    # =================================================================
    c = 343.0  # Speed of sound [m/s]
    f = 220.0  # Frequency [Hz]
    k_phys = 2*numpy.pi*f/c
    L_ref = 1.0  # Reference length [m]
    k = k_phys * L_ref
    omega = k
    print(f"Frequency = {f} Hz  →  k = {k:.3f} rad/unit")

    # =================================================================
    # PDE COEFFICIENTS
    # =================================================================
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # Sources (autoroute en haut)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob,
                                               amplitude=2.0, n_sources=6)
    # =================================================================
    # INITIAL ROBIN CONDITION: "Bare wall"
    # =================================================================
    alpha_rob[:, :] = -omega * 1j

    # =================================================================
    # INITIAL DENSITY χ ON FRACTAL
    # =================================================================
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # =================================================================
    # ABSORBING MATERIAL: α(ω) for porous material
    # =================================================================
    # For ISOREL at 125 Hz (from Chapter 3 / Figure 3.10)
    # These are example values - adjust based on your material
    Alpha = 8.950829344347426 - 1.0129061485319578E1*1j     
    alpha_rob = Alpha * chi

    # =================================================================
    # OPTIMIZATION PARAMETERS
    # =================================================================
    mask_R = (domain_omega == _env.NODE_ROBIN)
    S = numpy.sum(mask_R)
    
    # TUNABLE PARAMETERS:
    # -------------------
    V_obj = 0.4  # Target volume fraction β = 40%
    V_0 = V_obj * S  # Target total volume
    zeta0 = 0.7  # Initial step size (TUNE THIS)
    mu1 = 1e-10  # Volume penalty (TUNE THIS)
    max_iter = 100  # Maximum iterations
    delta = 5e-5  # Convergence tolerance
    
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
    
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()
    
    J0 = your_compute_objective_function(domain_omega, u0, spacestep, mu1, V_0, chi=chi0)
    print(f"Uncontrolled energy J₀ = {J0:.6e}")

    # =================================================================
    # OPTIMIZATION
    # =================================================================
    print(f"\n{'='*60}")
    print("STARTING OPTIMIZATION")
    print(f"{'='*60}")
    
    chi, energy, u, grad = your_optimization_procedure(
        domain_omega, spacestep, omega,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, zeta0, chi, V_obj, mu1, V_0,
        max_iter=max_iter, delta=delta
    )

    chin = chi.copy()
    un = u.copy()

    # Save relaxed solution
    chi_relaxed = chi.copy()
    u_relaxed = u.copy()

    # =================================================================
    # BINARY PROJECTION
    # =================================================================
    print(f"\n{'='*60}")
    print("BINARY PROJECTION")
    print(f"{'='*60}")
    
    mask_R = (domain_omega == _env.NODE_ROBIN)
    chi_bin = project_to_binary(chi_relaxed, mask_R, V_obj)
    alpha_rob_bin = Alpha * chi_bin

    # Recompute field for binary χ
    u_bin = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                    f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir,
                                    beta_neu, beta_rob, alpha_rob_bin)
    
    J_bin = your_compute_objective_function(domain_omega, u_bin, spacestep, mu1, V_0, chi=chi_bin)
    beta_bin = numpy.mean(chi_bin[mask_R])
    
    print(f"Binary χ: β = {beta_bin:.4f}, J = {J_bin:.6e}")

    # =================================================================
    # PLOTS
    # =================================================================
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(u_bin, chi_bin)
    err = u_bin - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Uncontrolled energy:  J₀ = {J0:.6e}")
    print(f"Relaxed energy:       J  = {your_compute_objective_function(domain_omega, u_relaxed, spacestep, mu1, V_0, chi=chi_relaxed):.6e}")
    print(f"Binary energy:        J  = {J_bin:.6e}")
    print(f"Improvement:          ΔJ/J₀ = {(J0 - J_bin)/J0:.2%}")
    print(f"Final β (relaxed):    {numpy.mean(chi_relaxed[mask_R]):.4f}")
    print(f"Final β (binary):     {beta_bin:.4f}")
    print(f"Target β:             {V_obj:.4f}")

    print('\nDone.')