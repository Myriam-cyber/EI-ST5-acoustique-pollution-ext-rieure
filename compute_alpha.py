# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import scipy
from scipy.optimize import minimize
import scipy.io

MATERIALS = {
    "MELAMINE": {"phi": 0.99, "sigma": 14000.0, "alpha_h": 1.02},
    "BIRCH": {"phi": 0.529, "sigma": 151429.0, "alpha_h": 1.37},
    "LAINE_ROCHE": {"phi": 0.98, "sigma": 30000.0, "alpha_h": 1.2},
    "LAINE_VERRE": {"phi": 0.95, "sigma": 40000.0, "alpha_h": 1.3},
    "BETON": {"phi": 0.45, "sigma": 5000.0, "alpha_h": 2.2},
}


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return numpy.array([numpy.real(z), numpy.imag(z)])


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        # .. todo: deepcopy here if returning objects
        return self.memo[args]


def compute_alpha(omega, material):
    """
    .. warning: $w = 2 \pi f$
    w is called circular frequency
    f is called frequency
    """

    #Birch LT
    #phi = 0.529  # porosity
    #gamma_p = 7.0 / 5.0
    #sigma = 151429.0  # resitivity
    #rho_0 = 1.2
    #alpha_h = 1.37  # tortuosity
    #c_0 = 340.0
    
    # melamine foam 
    #phi = 0.99  # porosity
    #gamma_p = 7.0 / 5.0
    #sigma = 14000.0  # resitivity
    #rho_0 = 1.2
    #alpha_h = 1.02  # tortuosity
    #c_0 = 340.0
    
    # laine de verre (source : https://backend.orbit.dtu.dk/ws/portalfiles/portal/337029887/ForumAcusticum_Compilation_1.pdf)
    #phi = 0.98  # porosity
    #gamma_p = 7.0 / 5.0
    #sigma = 4.0e4  # resitivity
    #rho_0 = 1.2
    #alpha_h = 1.3  # tortuosity
    #c_0 = 340.0
    
    # laine de roche (source : https://scispace.com/pdf/characterizing-modelling-and-optimizing-the-sound-absorption-1p9oyra5mr.pdf)
    #phi = 0.98  # porosity
    #gamma_p = 7.0 / 5.0
    #sigma = 30000  # resitivity
    #rho_0 = 1.2
    #alpha_h = 1.2  # tortuosity
    #c_0 = 340.0
    
    phi = MATERIALS[material]["phi"]
    sigma = MATERIALS[material]["sigma"]
    alpha_h = MATERIALS[material]["alpha_h"]
    gamma_p = 7.0/5.0
    rho_0 = 1.2
    c_0 = 340.0



    
    
    # parameters of the geometry
    L = 0.01

    # parameters of the mesh
    resolution = 12  # := number of elements along L

    # parameters of the material (cont.)
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)

    ksi_volume = phi * gamma_p / (c_0 ** 2)
    a_volume = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)
    mu_volume = phi / alpha_h
    k2_volume = (1.0 / mu_volume) * ((omega ** 2) / (c_0 ** 2)) * (ksi_volume + 1j * a_volume / omega)
    print(k2_volume)

    # parameters of the objective function
    A = 1.0
    B = 1.0

    # defining k, omega and alpha dependant parameters' functions
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return numpy.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return numpy.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = numpy.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / numpy.sqrt(2.0)) * numpy.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / numpy.sqrt(2.0)) * numpy.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

    @Memoize
    def g(y):
        # ..warning: not validated ***********************
        return 1.0

    @Memoize
    def g_k(k):
        # ..warning: not validated ***********************
        if k == 0:
            return 1.0
        else:
            return 0.0

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * numpy.exp(-lambda_0(k, omega) * L) \
                + (lambda_0(k, omega) * mu_0 + x) * numpy.exp(lambda_0(k, omega) * L))

    @Memoize
    def chi(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha, omega):
        expm = numpy.exp(-2.0 * lambda_0(k, omega) * L)
        expp = numpy.exp(+2.0 * lambda_0(k, omega) * L)

        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return ((A + B * (numpy.abs(k) ** 2)) \
                    * ( \
                                (1.0 / (2.0 * lambda_0(k, omega))) \
                                * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                   + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                                + 2 * L * numpy.real(chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega)))) \
                    + B * numpy.abs(lambda_0(k, omega)) / 2.0 * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                                                 + (numpy.abs(eta(k, alpha, omega)) ** 2) * (
                                                                             expp - 1.0)) \
                    - 2 * B * (lambda_0(k, omega) ** 2) * L * numpy.real(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega))))
        else:
            return ((A + B * (numpy.abs(k) ** 2)) * (L \
                                                     * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (
                                numpy.abs(eta(k, alpha, omega)) ** 2)) \
                                                     + complex(0.0, 1.0) * (1.0 / lambda_0(k, omega)) * numpy.imag(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                          * (1.0 - expm))))) + B * L * (
                               numpy.abs(lambda_0(k, omega)) ** 2) \
                   * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (numpy.abs(eta(k, alpha, omega)) ** 2)) \
                   + complex(0.0, 1.0) * B * lambda_0(k, omega) * numpy.imag(
                chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                  * (1.0 - expm)))

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * numpy.pi / L
                s += e_k(k, alpha, omega)
            return s

        return sum_func

    @Memoize
    def alpha(omega):
        alpha_0 = numpy.array(complex(40.0, -40.0))
        temp = real_to_complex(minimize(lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))), complex_to_real(alpha_0), tol=1e-4).x)
        print(temp, "------", "je suis temp")
        return temp
    # ici on peut essayer d'autres methodes de minimisation, pour optimiser les valeurs de alpha

    @Memoize
    def error(alpha, omega):
        temp = numpy.real(sum_e_k(omega)(alpha))
        return temp

    temp_alpha = alpha(omega)
    temp_error = error(temp_alpha, omega)

    return temp_alpha, temp_error


def run_compute_alpha(material):
    print('Computing alpha...')
    numb_omega = 1000
    omegas = numpy.linspace(2.0 * numpy.pi, 2.0 * numpy.pi * 1000, num=numb_omega)

    
    temp = [compute_alpha(omega, material=material) for omega in omegas]
    alphas, errors = map(list, zip(*temp))
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)

    frequencies = omegas / (2.0 * numpy.pi)  # <<< ajout conversion en Hz

    print('Writing alpha...')
    scipy.io.mmwrite('dta_omega_' + str(material) + '.mtx', omegas.reshape(len(omegas),1))
    scipy.io.mmwrite('dta_freq_' + str(material) + '.mtx', frequencies.reshape(len(frequencies),1))  # <<< nouveau
    scipy.io.mmwrite('dta_alpha_' + str(material) + '.mtx', alphas.reshape(len(alphas),1), field='complex')
    scipy.io.mmwrite('dta_error_' + str(material) + '.mtx', errors.reshape(len(errors),1))



def run_plot_alpha(material):
    print('Reading alpha...')

    # Read frequency (Hz)
    freqs = scipy.io.mmread('dta_freq_' + str(material) + '.mtx')
    freqs = freqs.reshape(freqs.shape[0])

    # Read alpha and error
    alphas = scipy.io.mmread('dta_alpha_' + str(material) + '.mtx')
    alphas = alphas.reshape(alphas.shape[0])
    errors = scipy.io.mmread('dta_error_' + str(material) + '.mtx')
    errors = errors.reshape(errors.shape[0])

    # Compute ratio Re(alpha)/Im(alpha)
    ratio = numpy.real(alphas) / numpy.imag(alphas)

    # ---- Plot ratio ----
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(freqs, ratio)
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    matplotlib.pyplot.ylabel('Re(alpha) / Im(alpha)')
    matplotlib.pyplot.title('Ratio Re(alpha)/Im(alpha) vs Frequency')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig('fig_ratio_alpha_freq_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    #print("Saved ratio figure: fig_ratio_alpha_freq_" + str(material) + ".jpg")
    
def plot_all_materials(materials):
    colors = {
        "MELAMINE": "blue",
        "BIRCH": "green",
        "LAINE_ROCHE": "red",
        "LAINE_VERRE": "purple",
        "BETON": "orange"
    }

    fig = matplotlib.pyplot.figure()
    
    for mat in materials:
        freqs = scipy.io.mmread(f'dta_freq_{mat}.mtx').reshape(-1)
        alphas = scipy.io.mmread(f'dta_alpha_{mat}.mtx').reshape(-1)

        ratio = abs(numpy.real(alphas)) / abs(numpy.imag(alphas))

        matplotlib.pyplot.plot(freqs, ratio, label=mat, color=colors[mat])

    matplotlib.pyplot.xlabel("Frequency (Hz)")
    matplotlib.pyplot.ylabel("Re(alpha)/Im(alpha)")
    matplotlib.pyplot.title("Comparison of mod(Re(alpha)/Im(alpha)) for porous materials")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig("compare_ratio_alpha_materials.jpg")
    matplotlib.pyplot.close(fig)
    print("Saved: compare_ratio_alpha_materials.jpg")
    

def plot_real_all_materials(materials):
    colors = {
        "MELAMINE": "blue",
        "BIRCH": "green",
        "LAINE_ROCHE": "red",
        "LAINE_VERRE": "purple",
        "BETON": "orange"
    }

    fig = matplotlib.pyplot.figure()

    for mat in materials:
        freqs = scipy.io.mmread(f'dta_freq_{mat}.mtx').reshape(-1)
        alphas = scipy.io.mmread(f'dta_alpha_{mat}.mtx').reshape(-1)

        real_alpha = numpy.real(alphas)

        matplotlib.pyplot.plot(freqs, real_alpha, label=mat, color=colors[mat])

    matplotlib.pyplot.xlabel("Frequency (Hz)")
    matplotlib.pyplot.ylabel("Re(α)")
    matplotlib.pyplot.title("Real part of α vs Frequency for all materials")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig("compare_real_alpha_materials.jpg")
    matplotlib.pyplot.close(fig)
    print("Saved: compare_real_alpha_materials.jpg")


def plot_imag_all_materials(materials):
    colors = {
        "MELAMINE": "blue",
        "BIRCH": "green",
        "LAINE_ROCHE": "red",
        "LAINE_VERRE": "purple",
        "BETON": "orange"
    }

    fig = matplotlib.pyplot.figure()

    for mat in materials:
        freqs = scipy.io.mmread(f'dta_freq_{mat}.mtx').reshape(-1)
        alphas = scipy.io.mmread(f'dta_alpha_{mat}.mtx').reshape(-1)

        imag_alpha = numpy.imag(alphas)

        matplotlib.pyplot.plot(freqs, imag_alpha, label=mat, color=colors[mat])

    matplotlib.pyplot.xlabel("Frequency (Hz)")
    matplotlib.pyplot.ylabel("Im(α)")
    matplotlib.pyplot.title("Imaginary part of α vs Frequency for all materials")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig("compare_imag_alpha_materials.jpg")
    matplotlib.pyplot.close(fig)
    print("Saved: compare_imag_alpha_materials.jpg")



def run():
    materials = ["MELAMINE", "BIRCH", "LAINE_ROCHE", "LAINE_VERRE", "BETON"]

    for mat in materials:
        run_compute_alpha(mat)

    # New plots
    plot_real_all_materials(materials)
    plot_imag_all_materials(materials)

    
    # plot_all_materials(materials)




if __name__ == '__main__':
    run()
    print('End.')