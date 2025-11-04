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

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        print('2. computing solution of adjoint problem, i.e., p')
        print('3. computing objective function, i.e., energy')
        print('4. computing parametric gradient')
        while ene >= energy[k] and mu > 10 ** -5:
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            p = processing.solve_adjoint(domain_omega, spacestep, omega, u,  beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            A = -(chi*Alpha*u*p).real
            J = numpy.sum(A) * (spacestep**2)
            chi = chi - mu*J
            print('    a. computing gradient descent')
            print('    b. computing projected gradient')
            print('    c. computing solution of Helmholtz problem, i.e., u')
            print('    d. computing objective function, i.e., energy (E)')
            ene = compute_objective_function(domain_omega, u, spacestep, mu1, V_0)
            energy[k] = ene
            if bool_a:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased is the energy increased
                mu = mu / 2
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy, u, grad


def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0):
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

    # 1. Calcul de l'intégrale (Energie) : \int_{domain_omega}||u||^2
    # Dans le cas discret : Somme sur le domaine de ||u_{i,j}||^2 * (spacestep)^2

    # Calcul de ||u||^2 (module au carré de u)
    # numpy.abs(u)**2 est équivalent à u * numpy.conjugate(u)
    # L'énergie physique est généralement liée à la partie réelle de cette intégrale.
    # Pour l'optimisation, on utilise le carré du module.
    modulus_squared = numpy.abs(u)**2

    # L'opérateur * (multiplication terme à terme) est nécessaire pour l'intégrale discrète.
    # L'intégration se fait uniquement sur les nœuds appartenant au domaine (domain_omega != 0)
    # Cependant, u n'est non nul que sur le domaine, donc une simple somme suffit souvent
    # si u est bien défini. Si on veut être rigoureux :
    # On met à zéro les termes en dehors du domaine (où domain_omega est 0)
    # La variable _env.NODE_DIR/NEU/ROBIN/INTERIOR sont non nulles pour les nœuds
    # faisant partie du domaine.

    # On suppose que u est déjà nul en dehors du domaine d'intérêt (comme c'est
    # souvent le cas après solve_helmholtz). Sinon, il faudrait masquer.
    # Dans le cas où u est seulement défini sur le domaine (i.e. u.shape = domain_omega.shape)
    # on utilise u.
    
    # Approximation par somme de Riemann
    integral_u_squared = numpy.sum(modulus_squared) * (spacestep**2)
    # On prend la partie réelle de la somme (important si u est complexe, même si
    # numpy.abs(u)**2 est réel) :
    real_integral_u_squared = numpy.real(integral_u_squared)

    # 2. Calcul du volume du domaine: Vol(domain_omega)
    # Le volume discret est la somme des "cellules" multipliée par (spacestep)^2
    # On compte le nombre de nœuds dans le domaine (là où domain_omega est non nul).
    # Tous les nœuds dont la valeur est > 0 font partie du domaine.
    volume_nodes = numpy.sum(domain_omega > 0)
    Vol_omega = volume_nodes * (spacestep**2)

    # 3. Calcul de l'énergie J(u,domain_omega)
    # J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)
    
    # Le premier terme (énergie acoustique/électromagnétique)
    energy_term = real_integral_u_squared
    
    # Le second terme (contrainte de volume)
    volume_constraint_term = mu1 * (Vol_omega - V_0)

    energy = energy_term + volume_constraint_term

    return energy


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    # chi, energy, u, grad = your_optimization_procedure(...)
    #chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
    # --- en of optimization

    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')







