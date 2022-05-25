import numpy as np
from scipy.optimize import curve_fit

beta = -2.5 #-2.505
nu_0 = 100

def EDGES_polynomial(nu, coeffs):
    if np.isscalar(nu):
        nu = np.array([nu])
    if nu.ndim == 1:
        nu.shape = (-1, 1)
    coeffs = np.array([coeffs])
    n = coeffs.shape[-1]
    t = (nu/nu_0)**beta * coeffs * (nu/nu_0)**np.arange(n).reshape(1, -1)
    return t.sum(axis=1)

# for n= 5
def EP_fit5(nu, a0, a1, a2, a3, a4):
    coeffs = [a0, a1, a2, a3, a4]
    return EDGES_polynomial(nu, coeffs)

# for n= 6
def EP_fit6(nu, a0, a1, a2, a3, a4, a5):
    coeffs = [a0, a1, a2, a3, a4, a5]
    return EDGES_polynomial(nu, coeffs)

# for n= 7
def EP_fit7(nu, a0, a1, a2, a3, a4, a5, a6):
    coeffs = [a0, a1, a2, a3, a4, a5, a6]
    return EDGES_polynomial(nu, coeffs)

def fit_EP(freqs, temps, N=6, p0=None):
    d = {"5": EP_fit5, "6": EP_fit6, "7": EP_fit7}
    EP_fit = d[str(N)]
    coeff_opt, pcov = curve_fit(EP_fit, freqs, temps, p0=p0)
    return coeff_opt, pcov


def linlog(nu, coeffs):
    if np.isscalar(nu):
        nu = np.array([nu])
    if nu.ndim == 1:
        nu.shape = (-1, 1)
    coeffs = np.array([coeffs])
    n = coeffs.shape[-1]
    t = (nu/nu_0)**beta * coeffs *(np.log(nu/nu_0))**np.arange(n).reshape(1, -1)
    return t.sum(axis=1)

# for n= 5
def ll_fit5(nu, a0, a1, a2, a3, a4):
    coeffs = [a0, a1, a2, a3, a4]
    return linlog(nu, coeffs)

# for n= 6
def ll_fit6(nu, a0, a1, a2, a3, a4, a5):
    coeffs = [a0, a1, a2, a3, a4, a5]
    return linlog(nu, coeffs)

# for n= 7
def ll_fit7(nu, a0, a1, a2, a3, a4, a5, a6):
    coeffs = [a0, a1, a2, a3, a4, a5, a6]
    return linlog(nu, coeffs)

def fit_ll(freqs, temps, N=6, p0=None):
    d = {"5": ll_fit5, "6": ll_fit6, "7": ll_fit7}
    ll_fit = d[str(N)]
    coeff_opt, pcov = curve_fit(ll_fit, freqs, temps, p0=p0)
    return coeff_opt, pcov

def add_cax(fig, ax, pad=0.01, width=0.02):
    """
    Adds colorbar axis next to existing axis in figure.
    pad = distance to axis
    width = width of cbar 
    """
    pos = [
           ax.get_position().x1+pad,
           ax.get_position().y0,
           width,
           ax.get_position().height
          ]
    cax = fig.add_axes(pos)
    return cax
