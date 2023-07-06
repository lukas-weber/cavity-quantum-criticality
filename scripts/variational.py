import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import json

from juliacall import Main as jl

jl.include("downfolded_peierls.jl")

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def exact_coupling(omega, g, max_photons):
    gen = jl.DownfoldedPeierlsGenerator(jl.Mode(omega, g, max_photons), tolerance=1e-9)
    return np.array(jl.matrix(gen))


def approx_coupling(omega, g, max_photons):
    n = np.arange(max_photons)

    Jdiag = 1 + g**2 * (-1 - 2 * n + (n + 1) / (1 + omega) + n / (1 - omega))
    Joffdiag = (
        g**2
        * ((n + 2) * (n + 1)) ** 0.5
        * (1 / (1 - omega**2) - 0.5 * (1 + 1 / (1 - 4 * omega**2)))
    )
    print("Joffdiag", Joffdiag[0] * 32**2 / 2**0.5 / 2)

    return sps.diags(
        [Joffdiag[:-2], Jdiag, Joffdiag[:-2]],
        [-2, 0, 2],
        shape=(max_photons, max_photons),
    )


def solve_J(
    g0,
    N,
    omega,
    U,
    spin_energy,
    max_photons,
    bond_angles=[0],
    jgen=exact_coupling,
):
    nphot = np.diag(np.arange(max_photons))
    H = (
        sum(
            jgen(omega, g0 * np.cos(phi) / N**0.5, max_photons) for phi in bond_angles
        )
        * spin_energy
        * N
        + U * omega * nphot
    )

    H[np.abs(H) < 1e-7] = 0
    H = sps.csc_matrix(H)

    E, psi = spsl.eigsh(H, k=1, which="SA")

    return E.min() / N, psi[:, 0]


def calc_energy_renormalization(g0, max_photons, omega, U, N, Es):
    print(g0)
    if g0 == 0:
        g0 = 1e-8
    Emin = np.array([solve_J(g0, N, omega, U, E, max_photons)[0] for E in Es])
    return Emin


def energy_renormalization(N, g0s, U, omega, max_photons):
    Es = np.linspace(-1, 0, 30)

    res = [
        calc_energy_renormalization(
            g0=g0, max_photons=max_photons, omega=omega, U=U, N=N, Es=Es
        )
        for g0 in g0s
    ]

    return res, Es


def calc_occupation(g0, max_photons, omega, U, E, invNs):
    print(g0)
    r = [solve_J(g0, 1 / iN, omega, U, E, max_photons)[1] for iN in invNs]
    nmin = [np.sum(np.arange(max_photons) * psi**2) for psi in r]
    return nmin


def occupation_scaling(U, omega, g0s, max_photons):
    E = -1
    Ns = np.concatenate([[0.001, 0.002, 0.005], np.linspace(0.01, 0.1, 17)])

    res = [
        calc_occupation(g0=g0, max_photons=max_photons, omega=omega, U=U, E=E, invNs=Ns)
        for g0 in g0s
    ]

    return res, Ns


def calculate_data(U, omega, g0s, N, max_photons):
    energies_renormalized, energies = energy_renormalization(
        N=N, g0s=g0s, U=U, omega=omega, max_photons=max_photons
    )
    occupations, invNs = occupation_scaling(
        g0s=g0s, U=U, omega=omega, max_photons=max_photons
    )

    data = [
        {
            "g0": g0,
            "energies_renormalized": Eren.tolist(),
            "energies": energies.tolist(),
            "occupations": occupation,
            "invNs": invNs.tolist(),
        }
        for g0, Eren, occupation in zip(g0s, energies_renormalized, occupations)
    ]

    with open("../data/variational.json", "w") as f:
        json.dump(data, f, indent=1)


def fig_variational(U, omega, N):
    fig, axs = plt.subplots(1, 2)

    with open("../data/variational.json", "r") as f:
        data = json.load(f)

    for i, d in enumerate(data):
        g0 = d["g0"]
        axs[0].plot(
            d["energies"],
            d["energies_renormalized"],
            "-",
            color=colors[i],
            label=f"$\\lambda={g0:.2g}$",
        )

    axs[0].set_ylabel("$E$")
    axs[0].set_xlabel("$E_\\mathrm{S}$")
    axs[0].legend()

    for i, d in enumerate(data):
        g0 = d["g0"]
        axs[1].plot(
            d["invNs"],
            np.array(d["occupations"]) * np.array(d["invNs"]),
            "-",
            label=f"$\\lambda={g0:.3g}$",
            color=colors[i],
        )
    axs[1].set_ylabel("$n_\\mathrm{ph}/N$")
    axs[1].set_xlabel("$1/N$")

    axs[0].text(
        0.05,
        0.98,
        f"(a) $N={N}$",
        verticalalignment="top",
        transform=axs[0].transAxes,
    )
    axs[1].text(
        0.2,
        0.98,
        "(b) $E_\\mathrm{S}=-1$",
        verticalalignment="top",
        transform=axs[1].transAxes,
    )
    plt.tight_layout(pad=0.1)
    plt.savefig("../plots/variational.pdf")
    plt.show()


if __name__ == "__main__":
    U = 200
    omega = 0.49
    g0s = [0, 1, 2, 3]

    N = 100
    max_photons = 200

    # calculate_data(U=U, omega=omega, g0s=g0s, N=N, max_photons=max_photons)
    fig_variational(U=U, omega=omega, N=N)
