from juliacall import Main
import numpy as np

Main.include("downfolded_peierls.jl")

JD = 1.6433


def lattice_weights(lattice):
    if lattice == "honeycomb":
        Js = [1, 1, 1, JD]
        gs = np.array([1.0, np.cos(2 * np.pi / 3), np.cos(-2 * np.pi / 3), 0])
        weights = [2.0, 2.0, 2.0, 1.0]
    elif lattice == "square":
        Js = [1, 1]
        gs = np.array([1.0, 0.0])
        weights = [1, 1]
    return Js, gs, weights


# omega is Omega/U g is g0/L
def sw_correction(lattice, omega, g, func):
    Js, gs, weights = lattice_weights(lattice)
    return np.array(
        Main.sw_correction(omega=omega, Js=Js, gs=g * gs, weights=weights, func=func)
    )


def photonic(lattice, energy, U, N, omega, g, func):
    return -energy * N / U * sw_correction(lattice, omega / U, g, func)


def safm(lattice, energy, U, omega, g):
    return 2 * energy / U * sw_correction(lattice, omega / U, g, func=lambda n: n > 0)
