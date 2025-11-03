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


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, mu1, V_0):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can it up to 1).
    """

    (M, N) = numpy.shape(domain_omega)
    numb_iter = 50
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)

    # solve initial state
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    J = your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=chi)
    energy[0] = J

    mask_R = (domain_omega == _env.NODE_ROBIN)

    for k in range(1, numb_iter+1):
        print('---- iteration number = ', k)

        # --- gradient "proxy" sur la frontière : pousse chi là où |u|^2 est grand
        grad = numpy.zeros_like(chi, dtype=numpy.float64)
        grad_R = numpy.abs(u[mask_R])**2
        # normalisation évite des pas trop gros
        if grad_R.size > 0:
            gmax = grad_R.max()
            if gmax > 0:
                grad[mask_R] = grad_R / gmax
            else:
                grad[mask_R] = 0.0

        # direction de descente: diminuer l'énergie -> chi_new = chi - mu * grad
        chi_trial = chi.copy()
        chi_trial[mask_R] = numpy.clip(chi[mask_R] - mu * grad[mask_R], 0.0, 1.0)

        # volume (optionnel) : recentre pour viser V_obj (sur Robin)
        if V_obj is not None and V_obj > 0:
            S = max(1, numpy.sum(mask_R))
            avg = numpy.sum(chi_trial[mask_R]) / S
            shift = avg - V_obj
            chi_trial[mask_R] = numpy.clip(chi_trial[mask_R] - 0.1*shift, 0.0, 1.0)

        # mettre à jour alpha_rob = Alpha * chi_trial pour résoudre u_trial
        alpha_rob_trial = Alpha * chi_trial

        # résoudre Helmholtz pour la densité d’essai
        u_trial = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                             beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_trial)
        J_trial = your_compute_objective_function(domain_omega, u_trial, spacestep, mu1, V_0, chi=chi_trial)

        # ligne de descente adaptative
        if J_trial < J:
            # succès: accepte, augmente légèrement mu
            chi = chi_trial
            alpha_rob = alpha_rob_trial
            u = u_trial
            J = J_trial
            mu = mu * 1.1
        else:
            # échec: diminue mu et ré-essaie depuis le même k avec pas plus petit
            mu = mu / 2.0
            if mu < 1e-5:
                print("Pas trop petit, arrêt anticipé.")
                energy[k:] = J
                break
            # ne pas incrémenter k artificiellement; continue avec pas réduit
            # mais on doit enregistrer l'énergie actuelle pour cet index
        energy[k] = J

    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy, u, grad


def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        mu1: float, it is the constant that defines the importance of the volume
        constraint;
        V_0: float, it is a reference volume.
    """
    """
    J = sum_{i,j in domaine} |u|^2 * spacestep^2  +  mu1 * (Vol(chi) - V_0)
    Vol(chi) = moyenne/somme de chi sur la frontière Robin (au choix, mais sois cohérent).
    """
    # énergie du champ dans le domaine intérieur
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    E = numpy.sum(numpy.abs(u[mask_dom])**2) * (spacestep**2)

    # terme de volume (si chi fourni)
    if chi is not None:
        mask_R = (domain_omega == _env.NODE_ROBIN)
        Vol = numpy.sum(chi[mask_R])  # ou /S si tu préfères une moyenne
        J = E + mu1 * (Vol - V_0)
    else:
        J = E
    return float(numpy.real(J))


if __name__ == '__main__':

    N = 50
    M = 2 * N
    level = 0
    spacestep = 1.0 / 10

    # --- fréquence 125 Hz
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

    # --- onde plane en haut (bruit voiture)
    f_dir[:, :] = 0.0
    theta_deg = 0.0
    theta = numpy.deg2rad(theta_deg)
    x_coords = numpy.arange(N) * spacestep
    phase_x = numpy.exp(1j * k * numpy.sin(theta) * x_coords)
    f_dir[0, 0:N] = phase_x

    # --- condition Robin de base
    alpha_rob[:, :] = - omega * 1j

    # --- densité initiale
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # --- matériau absorbant
    Alpha = 6.311111794566079 - 6.67835254158675*1j
    alpha_rob = Alpha * chi

    # --- paramètres optimisation
    S = numpy.sum(domain_omega == _env.NODE_ROBIN)
    V_0 = 1.0
    V_obj = numpy.sum(chi) / S
    mu = 5.0
    mu1 = 1e-5

    # --- solution non contrôlée (pour comparer)
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # --- DESCENTE DE GRADIENT
    chi, energy, u, grad = your_optimization_procedure(
        domain_omega, spacestep, omega,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, mu, chi, V_obj, mu1, V_0
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
