import numpy as np
from scipy.optimize import curve_fit

beta = -2.5
nu_0 = 100

def EDGES_polynomial(nu, coeffs):
    t = (nu/nu_0)**beta
    terms = 0
    for n, coeff in enumerate(coeffs):
        terms += coeff * (nu/nu_0)**n
    t *= terms
    return t

# for n= 6
def EP_fit(nu, a0, a1, a2, a3, a4, a5):
    coeffs = [a0, a1, a2, a3, a4, a5]
    return EDGES_polynomial(nu, coeffs)

def fit_EP(freqs, temps, p0=None):
    coeff_opt, pcov = curve_fit(EP_fit, freqs, temps, p0=p0)
    return coeff_opt, pcov
