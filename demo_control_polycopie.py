# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions

def project_Uad_star(chi_tentative, mask_R, beta_target, tol=1e-6, max_iter=50):
    """
    Projection PU*_ad(β)(χ) = max(0, min(χ + ℓ, 1)) sur la frontière Robin,
    avec ℓ choisi pour que la moyenne sur Γ (Robin) soit β = beta_target.
    """
    vals = chi_tentative[mask_R]
    S = vals.size
    if S == 0:
        return chi_tentative

    # bornes initiales pour ℓ
    ell_min = -1.0
    ell_max = 1.0

    for _ in range(max_iter):
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


def compute_parametric_gradient(domain_omega, Alpha, u, p):
    """
    Gradient paramétrique g(x) ≈ Re(Alpha * p * u) sur Γ_Robin,
    en utilisant les valeurs d'intérieur voisines pour approximer la trace.

    Pour chaque nœud Robin (i,j), on prend la moyenne des voisins intérieurs
    (i±1,j), (i,j±1) et on calcule Re(Alpha * p_int * u_int).
    """
    (M, N) = numpy.shape(domain_omega)
    grad = numpy.zeros((M, N), dtype=numpy.float64)

    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                vals_u = []
                vals_p = []
                # voisins (haut, bas, gauche, droite)
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
                    grad[i, j] = numpy.real(Alpha * p_avg * u_avg)
                else:
                    grad[i, j] = 0.0

    return grad



def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    J(u,chi) = ∫_Ω |u|² dx + mu1 * (Vol(chi) - V_0),
    avec Vol(chi) la somme de chi sur la frontière Robin.
    """
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    E = numpy.sum(numpy.abs(u[mask_dom])**2) * (spacestep**2)

    if chi is not None:
        mask_R = (domain_omega == _env.NODE_ROBIN)
        Vol = numpy.sum(chi[mask_R])
        J = E + mu1 * (Vol - V_0)
    else:
        J = E

    return float(numpy.real(J))


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, zeta0, chi_init, V_obj, mu1, V_0,
                           max_iter=50, delta=1e-3):
    """
    Descente de gradient paramétrique robuste :
    - utilise Re(Alpha * p * u) sur Γ_R,
    - teste les 2 directions : d = -grad et d = +grad,
    - pour chaque direction, fait une mini recherche de pas par dichotomie,
    - choisit la meilleure (celle qui diminue le plus J).
    """

    (M, N) = numpy.shape(domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)

    numb_iter = max_iter
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)

    # χ_0 et alpha_rob initial
    chi = chi_init.copy()
    alpha_rob_curr = alpha_rob.copy()

    # --- état direct u(χ_0)
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob_curr)

    # énergie initiale
    J = your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=chi)
    energy[0] = J

    zeta = zeta0
    rel_tol = 1e-8  # tolérance relative sur J

    for n in range(numb_iter):
        print(f"---- iteration number = {n}, J = {J:.6e}, zeta = {zeta:.3e}")

        # 1) Problème adjoint p(χ_n)
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u,
                                     beta_pde, alpha_pde, alpha_dir,
                                     beta_neu, beta_rob, alpha_rob_curr)

        # 2) Gradient brut g_n(x) = Re(Alpha * p * u)|_Γ
        grad = compute_parametric_gradient(domain_omega, Alpha, u, p)

        best_J = J
        best_chi = chi
        best_u = u
        best_alpha_rob = alpha_rob_curr
        best_zeta = zeta
        found_better = False

        # On teste les deux directions : -grad (descente classique) et +grad (au cas où signe inversé)
        for direction in [-1.0, 1.0]:
            # recherche de pas pour cette direction
            zeta_trial_base = zeta
            for j in range(10):  # au plus 10 essais de pas
                zeta_trial = zeta_trial_base / (2.0**j)

                chi_tent = chi + direction * zeta_trial * grad
                chi_tent = project_Uad_star(chi_tent, mask_R, V_obj)
                alpha_rob_tent = Alpha * chi_tent

                u_tent = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                                    f, f_dir, f_neu, f_rob,
                                                    beta_pde, alpha_pde, alpha_dir,
                                                    beta_neu, beta_rob, alpha_rob_tent)
                J_tent = your_compute_objective_function(domain_omega, u_tent,
                                                         spacestep, mu1, V_0, chi=chi_tent)

                if J_tent < best_J * (1.0 - rel_tol):
                    # meilleure amélioration trouvée
                    best_J = J_tent
                    best_chi = chi_tent
                    best_u = u_tent
                    best_alpha_rob = alpha_rob_tent
                    best_zeta = zeta_trial
                    found_better = True

        if not found_better:
            print("  Aucun pas ne diminue J (même en changeant le signe du gradient). Arrêt.")
            energy[n+1:] = J
            break

        # Acceptation du meilleur pas
        chi_next = best_chi
        u_next = best_u
        alpha_rob_next = best_alpha_rob
        J_next = best_J

        # mise à jour du pas de base (on l'augmente un peu si ça marche)
        zeta = best_zeta + 0.001

        energy[n+1] = J_next

        diff = numpy.max(numpy.abs(chi_next[mask_R] - chi[mask_R]))
        print(f"  J_new = {J_next:.6e}, diff_chi = {diff:.3e}, new zeta = {zeta:.3e}")

        if diff < delta:
            print("  Convergence atteinte : variation de chi < delta.")
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

    print('end. computing solution of Helmholtz problem, i.e., u')
    return chi, energy, u, grad



if __name__ == '__main__':

    # --- géométrie
    N = 50
    M = 2 * N
    level = 0
    spacestep = 1.0 / N

    # --- physique : 125 Hz, c = 343 m/s
    c = 343.0
    f = 125.0
    k_phys = 2*numpy.pi*f/c
    L_ref = 1.0
    k = k_phys * L_ref
    omega = k
    print(f"Fréquence = {f} Hz  →  k = {k:.3f} rad/unité")

    # --- coefficients PDE
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # --- onde plane en haut (bruit autoroute)
    f_dir[:, :] = 0.0
    alpha_dir[:, :] = 1.0
    theta_deg = 0.0
    theta = numpy.deg2rad(theta_deg)
    x_coords = numpy.arange(N) * spacestep
    phase_x = numpy.exp(1j * k * numpy.sin(theta) * x_coords)
    f_dir[0, 0:N] = phase_x

    # --- condition Robin initiale "mur nu" (si tu veux)
    alpha_rob[:, :] = - omega * 1j

    # --- densité initiale chi sur la fractale
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # --- matériau absorbant dépendant de chi
    Alpha = 6.311111794566079 - 6.67835254158675*1j
    alpha_rob = Alpha * chi

    # --- paramètres optimisation
    mask_R = (domain_omega == _env.NODE_ROBIN)
    S = numpy.sum(mask_R)
    V_0 = 1.0
    V_obj = numpy.sum(chi[mask_R]) / S      # moyenne initiale (β)
    zeta0 = 0.5                             # pas initial ζ_0
    mu1 = 1e-5

    # --- solution non contrôlée
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # --- optimisation (descente de gradient paramétrique)
    chi, energy, u, grad = your_optimization_procedure(
        domain_omega, spacestep, omega,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, zeta0, chi, V_obj, mu1, V_0,
        max_iter=50, delta=1e-3
    )

    chin = chi.copy()
    un = u.copy()

    # --- plots
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')

