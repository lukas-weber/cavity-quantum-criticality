import mcextract
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import json
import sw_correction


def plot_obs(ax, mc, ed, obsname, ylabel, uncorrected_inset):
    max_photonss = mc.get_parameter("max_photons", unique=True)
    U = 200
    obs0 = mc.get_observable(obsname, filter=dict(g0=0, T=0.05))
    o0 = unp.uarray(obs0.mean, obs0.error).mean()
    E0 = mc.get_observable("Energy", filter=dict(g0=0, T=0.05))
    uE0 = unp.uarray(E0.mean, E0.error).mean()
    ax_ins = uncorrected_inset

    plot_ins = []

    markers = ["o", "^"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    for max_photons in max_photonss:
        Ts = mc.get_parameter("T", unique=True)
        g0s = mc.get_parameter("g0", unique=True)
        for i, g0 in enumerate(g0s[1:]):
            if g0 > 0:
                ed_omegas = np.array(ed[str(g0)]["omegas"])

                for a in [ax, ax_ins] if uncorrected_inset else [ax]:
                    a.plot(
                        ed_omegas / U,
                        ed[str(g0)][obsname],
                        color="black",
                        marker=markers[i],
                        label=f"{g0/4} Hubbard",
                    )
            for T in [Ts[0]]:
                cond = dict(T=T, g0=g0, max_photons=max_photons)
                omegas = mc.get_parameter("omega", filter=cond)
                obs = mc.get_observable(obsname, filter=cond)

                obsu = unp.uarray(obs.mean, obs.error)

                N = 4**2
                if obsname != "PhotonNum":
                    obsu /= o0

                if obsname != "Energy":
                    plot_ins.append(
                        ax_ins.errorbar(
                            omegas / U,
                            unp.nominal_values(obsu),
                            unp.std_devs(obsu),
                            color=colors[i],
                            label=f"{g0/4} Heisen",
                            marker=markers[i],
                            markerfacecolor="white",
                        )
                    )

                if obsname.endswith("Mag2"):
                    obsu += np.array(
                        [
                            sw_correction.safm(
                                "square",
                                energy=uE0,
                                omega=omega,
                                U=U,
                                g=g0 / 4,
                            )
                            for omega in omegas
                        ]
                    )
                elif obsname == "PhotonNum":
                    obsu += np.array(
                        [
                            sw_correction.photonic(
                                "square",
                                energy=uE0,
                                omega=omega,
                                g=g0 / 4,
                                N=N,
                                U=U,
                                func=lambda n: n,
                            )
                            for omega in omegas
                        ]
                    )
                ax.errorbar(
                    omegas / U,
                    unp.nominal_values(obsu),
                    unp.std_devs(obsu),
                    color=colors[i],
                    label=f"{g0/4} Heisen + SW",
                    marker=markers[i],
                )
    if uncorrected_inset:
        ax_ins.legend(handles=plot_ins, fontsize=7, borderpad=0)
    ax.set_ylabel(ylabel)


def fig_ed_comparison():
    fig, axs = plt.subplots(3, 1, figsize=(3.375, 4), sharex=True)

    mc = mcextract.MCArchive("../data/ed_comparison/mc.json")
    with open("../data/ed_comparison/ed.json", "r") as f:
        ed = json.load(f)

    axins1 = axs[1].inset_axes([0.15, 0.15, 0.4, 0.5])
    axins2 = axs[2].inset_axes([0.15, 0.35, 0.4, 0.5])
    axins1.tick_params(pad=2, labelsize=7)
    axins2.tick_params(pad=2, labelsize=7)
    axins1.set_ylim(0.999, 1.0001)
    axins2.set_ylim(-0.0005, 0.01)

    plot_obs(axs[0], mc, ed, "Energy", "$E/E_{\\lambda=0}$", uncorrected_inset=None)
    plot_obs(
        axs[1],
        mc,
        ed,
        "StagXStagYMag2",
        "$S^{\\mathrm{AFM}}/S^{\\mathrm{AFM}}_{\\lambda=0}$",
        uncorrected_inset=axins1,
    )
    plot_obs(
        axs[2],
        mc,
        ed,
        "PhotonNum",
        "$n_\\mathrm{ph}$",
        uncorrected_inset=axins2,
    )
    axs[2].set_xlabel("$\\Omega/U$")
    line = axs[0].legend(title="$\\lambda/L$")
    line.get_title().set_position([-20, 0])
    axs[1].set_ylim(0.9985, 1.0001)
    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.2)
    plt.savefig("../plots/ed_comparison.pdf")
    plt.show()


fig_ed_comparison()
