from loadleveller import mcextract
import numpy as np
import matplotlib.pyplot as plt
import variational
import uncertainties
import uncertainties.unumpy as unp
import sw_correction

mc = mcextract.MCArchive("../data/honeycomb_bilayer/critical.json")


def squeezed_state(r, max_photons=8):
    cs = np.array(
        [
            (np.math.factorial(2 * n)) ** 0.5
            / (2**n * np.math.factorial(n))
            * np.tanh(r) ** n
            / np.cosh(r) ** 0.5
            for n in range(max_photons // 2)
        ]
    )

    return np.abs(cs) ** 2


def plot_phot_dist(ax, g0, states):
    width = 0.8 / len(states)
    off = width * (np.arange(len(states)) - (len(states) - 1) / 2)
    for i, (JD, state) in enumerate(states.items()):
        n = np.arange(len(state))
        ax.bar(
            n + off[i],
            unp.nominal_values(state),
            width=0.8 * width,
            yerr=unp.std_devs(state),
            label=f"${JD}$",
        )


def plot_phot_num(ax):
    mc = mcextract.MCArchive("../data/honeycomb_bilayer/scan2.json")
    for L in [32, 64]:
        cond = dict(Lx=L, g0=1.5, max_photons=8)
        JDs = mc.get_parameter("JD", filter=cond)
        obs = mc.get_observable("PhotonNum", filter=cond)

        E = mc.get_observable("Energy", filter=cond)
        uE = unp.uarray(E.mean, E.error)
        U = mc.get_parameter("U", unique=True, filter=cond)[0]
        omega = mc.get_parameter("omega", unique=True, filter=cond)[0]
        g = mc.get_parameter("g", unique=True, filter=cond)[0]
        correction = sw_correction.photonic(
            "honeycomb",
            omega=omega,
            energy=uE,
            U=U,
            N=2 * L**2,
            g=g / 3**0.5,
            func=lambda n: n,
        )

        uobs = unp.uarray(obs.mean, obs.error)
        uobs += correction

        ax.errorbar(JDs, unp.nominal_values(uobs), unp.std_devs(uobs), label=f"${L}$")
    ax.set_ylim(0.019, None)
    leg = ax.legend(ncol=2, title="$L=$", fontsize=7, loc="lower center")

    leg._legend_box.align = "left"

    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlabel("$J_{\\mathrm{D}}/J$", fontsize=7, labelpad=1)
    ax.set_ylabel("$\\langle n_{\\mathrm{ph}} \\rangle$", labelpad=2, fontsize=7)


def fig_phot_dist(mc):
    fig, ax = plt.subplots(1, 1)
    JDs = mc.get_parameter("JD", unique=True)
    JDc = JDs[np.argmin(np.abs(JDs - 1.643))]
    g0 = 1.5
    omega = mc.get_parameter("omega", unique=True)[0]
    U = mc.get_parameter("U", unique=True)[0]
    max_photons = 8
    L = 32

    states = {}
    for JDn, JD in zip(["J", "J_\\mathrm{D}^c"], JDs[[0, 1]]):
        obs = mc.get_observable(
            "PhotonHist",
            filter=dict(Lx=L, g0=g0, max_photons=max_photons, JD=JD),
        )
        E = mc.get_observable(
            "Energy", filter=dict(Lx=L, g0=g0, max_photons=max_photons, JD=JD)
        )
        uE = uncertainties.ufloat(E.mean[0], E.error[0])
        correction = np.array(
            [
                sw_correction.photonic(
                    "honeycomb",
                    omega=omega,
                    energy=uE,
                    U=U,
                    N=2 * L**2,
                    g=g0 / L / 3**0.5,
                    func=lambda n: n == s,
                )
                for s in range(len(obs.mean[0]))
            ]
        )
        correction[0] = -np.sum(correction[1:])
        print(correction)
        states[JDn] = unp.uarray(obs.mean[0], obs.error[0]) + correction

    Emin, v = variational.solve_J(
        g0=g0 / 3**0.5,
        max_photons=max_photons,
        omega=omega / U,
        U=U,
        N=L**2,
        spin_energy=-1 / 2,
        bond_angles=[0, 2 * np.pi / 3, -2 * np.pi / 3],
    )
    n = np.arange(max_photons)
    v[1::2] = 0
    correction = np.array(
        [
            sw_correction.photonic(
                "honeycomb",
                omega=omega,
                energy=3 * Emin,
                U=U,
                N=2 * L**2,
                g=g0 / L / 3**0.5,
                func=lambda n: n == s,
            )
            for s in range(len(obs.mean[0]))
        ]
    )
    correction[0] = -np.sum(correction[1:])
    print(correction)
    states["\\infty"] = unp.uarray(v**2, 0) + correction

    plot_phot_dist(ax, g0, states=states)

    axins = fig.add_axes([0.73, 0.72, 0.25, 0.25])
    axins.axvline(JDc, color="black", lw=0.5)
    plot_phot_num(axins)
    ax.set_yscale("log")
    ax.legend(title="$J_{\\mathrm{D}}=$", borderpad=0, loc="lower right")
    ax.text(
        0.02,
        0.98,
        f"$\\lambda = {g0/3**0.5:.3g}$",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.set_ylim(1e-11, None)
    ax.set_xlabel("$n_{\\mathrm{ph}}$")
    ax.set_xticks(n)
    ax.set_ylabel("$P(n_{\\mathrm{ph}})$")
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(top=0.98)
    plt.savefig("../plots/honeycomb_photons.pdf")
    plt.show()


fig_phot_dist(mc)
