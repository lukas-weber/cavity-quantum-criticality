import mcextract
import numpy as np
import obslabels
import matplotlib.pyplot as plt
import scipy.optimize as spo
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import uncertainties
import uncertainties.unumpy as unp
import exponents

datadir = "../data/critical_scaling/"
mc_honey_noncrit = mcextract.MCArchive(datadir + "honeycomb_noncritical.json")
mc_honey = mcextract.MCArchive(datadir + "honeycomb_critical.json")
mc_square = mcextract.MCArchive(datadir + "columnar_dimer_critical.json")
JDs_honey = mc_honey.get_parameter("JD", unique=True)
JDs_square = mc_square.get_parameter("JD", unique=True)

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]
markers = ["o", "s", "^", "v", "<", ">", "d"]

photon_line_kwargs = {
    4: dict(ls=""),
    8: dict(ls=""),
    16: dict(ls="--"),
}

custom_lines = [
    Line2D(
        [0],
        [0],
        color="black",
        label=f"$n_{{\\mathrm{{ph}}}}^\\mathrm{{max}}={num}$",
        **args,
    )
    for num, args in photon_line_kwargs.items()
]

nu = exponents.nu.n
beta = exponents.beta.n


def plot_obs_ratio(
    ax,
    mc,
    obsname,
    g0s,
    max_photons=8,
    fit_exp=None,
    misccond={},
    skip=2,
    Lmax=None,
    bond_length=1,
    colors=colors,
    kwargs={"ls": ""},
    func=lambda Ls, obs, ref: obs / ref,
    markers=markers,
):
    handles = []

    ratios = []
    ratio_errs = []
    Lss = []

    for i, g0 in enumerate(g0s):
        Ls = mc.get_parameter(
            "Lx", filter=dict(max_photons=max_photons, g0=g0, **misccond)
        )
        obs = mc.get_observable(
            obsname, filter=dict(max_photons=max_photons, g0=g0, **misccond)
        )
        obs_ref = mc.get_observable(
            obsname, filter=dict(max_photons=max_photons, g0=0, **misccond)
        )

        obsu = unp.uarray(obs.mean, obs.error)
        obs_refu = unp.uarray(obs_ref.mean, obs_ref.error)
        # obs_ref = mc_ref.get_observable(obsname)
        #

        ratio = func(Ls, obsu, obs_refu)

        mask = Ls >= Ls[skip]
        if Lmax is not None:
            mask = np.logical_and(mask, Ls <= Lmax)

        line = ax.errorbar(
            1 / Ls[mask],
            unp.nominal_values(ratio)[mask],
            unp.std_devs(ratio)[mask],
            color=colors[i],
            marker=markers[i],
            label=f"$\\lambda={g0/bond_length:.3g}$",
            **kwargs,
        )
        handles.append(line)

        mask = np.logical_not(np.isnan(obs.mean / obs_ref.mean))

        mask[0:skip] = False
        ratio_mean = unp.nominal_values(ratio[mask])
        ratio_err = unp.std_devs(ratio[mask])
        Ls = Ls[mask]

        ratios = np.concatenate([ratios, ratio_mean])
        ratio_errs = np.concatenate([ratio_errs, ratio_err])
        Lss += [(g0, L) for L in Ls]

    Lss = np.array(Lss).T

    if fit_exp is not None:

        def extra(x, a, d):
            g, L = x
            return a * L ** (fit_exp[0]) * (g**4 + d * g**6) + 1

        popt, pcov = spo.curve_fit(
            extra,
            Lss,
            ratios,
            sigma=ratio_errs,
            absolute_sigma=True,
            maxfev=8000,
        )
        chisq = np.sum((ratios - extra(Lss, *popt)) ** 2 / ratio_errs**2) / (
            len(ratios) - len(popt)
        )
        perr = np.diag(pcov) ** 0.5
        popterr = np.array([uncertainties.ufloat(p, sp) for p, sp in zip(popt, perr)])
        print(" ".join("{:1uS}".format(p) for p in popterr), chisq)

        xx = np.logspace(np.log(Ls[0]) / np.log(10), 8, 200)
        if perr[1] < popt[1]:
            for g0 in g0s:
                (l,) = ax.plot(
                    1 / xx,
                    extra([g0, xx], *popt),
                    marker="",
                    label=f"$1+ c_\\lambda L^{{{fit_exp[1]}}}$",
                    color="black",
                )
            handles = [l] + handles
    ax.set_xlim(-0.001, None)
    return handles


def fig_strucfac(
    mc,
    obss,
    fit_exps,
    max_photons,
    title="",
    misccond={},
    skip=2,
    bond_length=1,
    xlim=(0, 0.09),
    savename=None,
):
    fig, axs = plt.subplots(3, 1, figsize=(3.375, 4), sharex=True)
    for i, (ax, (obsname, obslabel), fit_exp) in enumerate(zip(axs, obss, fit_exps)):
        panel = chr(ord("a") + i)
        handles = plot_obs_ratio(
            ax,
            mc,
            obsname,
            g0s=[0.5, 1, 1.5],
            fit_exp=fit_exp,
            max_photons=max_photons,
            misccond=misccond,
            skip=skip,
            bond_length=bond_length,
        )
        ax.set_ylabel("${0}/{0}_{{\\lambda=0}}$".format(obslabel))
        ax.set_xlim(xlim)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))

        if ax != axs[0]:
            handles = [handles[0]]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.035))
        ax.text(
            0.015,
            0.95,
            f"({panel})",
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
        )
    axs[-1].set_xlabel("$1/L$")
    ylim = axs[0].get_ylim()
    axs[0].set_ylim(ylim[0], 1 + (ylim[1] - 1) * 1.2)

    axs[0].text(
        0.97,
        0.94,
        title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs[0].transAxes,
    )
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(top=0.99)
    if savename:
        plt.savefig(f"../plots/{savename}.pdf")
    plt.show()


def plot_critical(
    ax,
    mc,
    obsname,
    max_photons=8,
    xlim=None,
    ylim=None,
    Ldim=0,
    bond_length=3**0.5,
    misccond={},
):
    handles = []

    cond = dict(max_photons=max_photons, **misccond)
    g0s = np.array(mc.get_parameter("g0", filter=cond, unique=True))
    for i, g0 in enumerate(g0s):
        Ls = mc.get_parameter("Lx", filter=dict(g0=g0, **cond))
        obs = mc.get_observable(obsname, filter=dict(g0=g0, **cond))
        obs.mean *= Ls**Ldim
        obs.error *= Ls**Ldim

        line = ax.errorbar(
            Ls,
            obs.mean,
            obs.error,
            color=colors[i],
            label=f"${g0/bond_length:.3g}$",
        )
        handles.append(line)
    ax.set_ylabel("${}$".format(obslabels.label(obsname)))
    ax.set_xlabel("$L$")
    ax.set_yscale("log")
    ax.set_xscale("log")


obsnames = [
    "PhotonNum",
    "PhotonNumVar",
    "StagUCMag2",
    "StagUCMagChi",
    "Energy",
]

obss_honey = [
    ("Energy", "E"),
    ("StagUCMag2", "S^{\\mathrm{AFM}}"),
    ("StagUCMagChi", "\\chi^{\\mathrm{AFM}}"),
]
obss_square = [
    ("Energy", "E"),
    ("StagXStagUCMag2", "S^{\\mathrm{AFM}}"),
    ("StagXStagUCMagChi", "\\chi^{\\mathrm{AFM}}"),
]


def fig_absolute():
    fig, axss = plt.subplots(2, 2, figsize=(3.375, 4))
    JD_labels = dict(zip(JDs_honey, ["J", "J_\\mathrm{D}^c", "3 J"]))

    misccond = {"JD": JDs_honey[1]}
    plot_critical(axss[0][0], mc_honey, "StagUCMag2", Ldim=2, misccond=misccond)
    axss[0][0].set_ylabel(ylabel="$S^\\mathrm{AFM}$")
    handles = plot_obs_ratio(
        axss[0][1],
        mc_honey,
        "StagUCMag2",
        g0s=[0.5, 1, 1.5],
        misccond=misccond,
        bond_length=3**0.5,
        colors=colors[1:],
        kwargs=dict(ls="-"),
        func=lambda Ls, obs, ref: Ls**2 * (obs - ref),
    )
    axss[0][1].axhline(0, color=colors[0])
    axss[0][1].set_ylabel("$S^\\mathrm{AFM}-S^\\mathrm{AFM}_{\\lambda=0}$")

    obsnames = ["StagUCMagChi", "StagUCMag2", "Energy"]

    for ax, JD, mc in zip(axss[1], JDs_honey[[1, 2]], [mc_honey, mc_honey_noncrit]):
        handles = []
        for obsname, marker in zip(obsnames, [0, 1, 2]):
            Lmax = 34 if mc == mc_honey_noncrit else None
            handles += plot_obs_ratio(
                ax,
                mc,
                obsname,
                g0s=[1.5],
                misccond=dict(JD=JD),
                bond_length=3**0.5,
                colors=colors[marker:],
                markers=markers[marker:],
                kwargs=dict(ls="-", markerfacecolor="white"),
                func=lambda Ls, obs, ref: unp.fabs(obs - ref)
                * (Ls**2 if obsname.endswith("Mag2") else 1),
                Lmax=Lmax,
            )
        axss[1, 1].legend(
            loc="lower left",
            ncol=1,
            handles=handles,
            title="$A=$",
            labels=["$\\chi^\\mathrm{AFM}$", "$S^\\mathrm{AFM}$", "$E$"],
        )
        ax.set_ylabel("$|\\Delta A|$")
        ax.set_yscale("log")

    axss[1, 0].set_ylim(1e-5, 4000)
    axss[0][1].axhline(0, color=colors[0])
    for ax in [axss[0, 1], axss[1, 0], axss[1, 1]]:
        ax.set_xlabel("$1/L$")
    axss[0][1].set_ylabel("$\\Delta S^\\mathrm{AFM}$")

    axss[1, 0].text(
        0.28,
        0.9,
        "$\\lambda=0.866$",
        va="top",
        fontsize=7,
        transform=axss[1, 0].transAxes,
    )
    axss[1, 1].text(
        0.28,
        0.9,
        "$\\lambda=0.866$",
        va="top",
        fontsize=7,
        transform=axss[1, 1].transAxes,
    )
    for i, (ax, JDi) in enumerate(zip(axss.flat, [1, 1, 1, 2])):
        ax.text(
            0.05,
            0.98,
            f"({chr(ord('a')+i)}) $J_\\mathrm{{D}} = {JD_labels[JDs_honey[JDi]]}$",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
        )
    axss[0][0].legend(title="$\\lambda=$", loc="lower right")
    axss[0][0].set_ylim(1, None)
    plt.tight_layout(pad=0.4)
    plt.savefig("../plots/honeycomb_absolute.pdf")
    plt.show()


fig_absolute()

fit_exps = [(-2, "-2"), (1 / nu - 2, "1/\\nu - d"), (1 / nu - 2, "1/\\nu - d")]
fig_strucfac(
    mc_honey,
    obss_honey,
    fit_exps=fit_exps,
    max_photons=8,
    skip=1,
    bond_length=3**0.5,
    title="$J_{\\mathrm{D}} = J_{\\mathrm{D}}^c$",
    misccond={"JD": JDs_honey[1]},
    savename="scaling_honeycomb",
)

fig_strucfac(
    mc_square,
    obss_square,
    fit_exps=fit_exps,
    skip=2,
    xlim=(0, 0.035),
    max_photons=8,
    title="columnar dimer, $J_\\mathrm{D} = J_\\mathrm{D}^c$",
    misccond={"phi": 0},
    bond_length=1,
    savename="scaling_columnar_dimer",
)
