import mcextract
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from matplotlib.lines import Line2D
from dataclasses import dataclass


mc = mcextract.MCArchive("../data/crossings/mc.json")

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
]
markers = [
    "s",
    "^",
    "o",
    "v",
    "<",
    ">",
    "d",
    "o",
    "s",
    "^",
]

Ls = mc.get_parameter("Lx", unique=True)
Ls = Ls[Ls <= 80]
Ls = Ls[Ls >= 12]

colorsets = [
    lambda i: plt.get_cmap("Blues")((i + 1) / (len(Ls))),
    lambda i: plt.get_cmap("Oranges")((i + 1) / (len(Ls))),
]

photon_line_kwargs = {
    4: dict(ls="-"),
    8: dict(ls="--"),
    16: dict(ls="--", dashes=[1, 1, 3, 1]),
}

JDc = 1.6433


def fit(x, y, xc, range):
    mask = np.abs(x - xc) < range
    popt = np.polyfit(x[mask], y[mask], deg=1)

    return popt


@dataclass
class Fit:
    popt: list[float]
    xc: float
    range: float

    def __call__(self, x):
        return np.polyval(self.popt, x)

    def plot_data(self):
        x = np.linspace(self.xc - self.range, self.xc + self.range, 50)
        return (x, self(x))


def calculate_fits(mc, obsname, cond, Ldim=0, nsamples=200):
    Ls = mc.get_parameter("Lx", unique=True, filter=cond)
    Ls = Ls[Ls <= 80]
    Ls = Ls[Ls >= 12]

    fitdata = {}

    for i, L in enumerate(Ls):
        JDs = mc.get_parameter("JD", filter=dict(Lx=L, **cond))
        obs = mc.get_observable(obsname, filter=dict(Lx=L, **cond))
        xc = 1.643
        fitrange = 0.03 / (L / 8)
        fitdata[L] = [
            Fit(
                popt=fit(JDs, obs.mean * L**Ldim, xc, fitrange),
                xc=xc,
                range=fitrange,
            )
        ]
        for i in range(nsamples):
            fitdata[L].append(
                Fit(
                    popt=fit(
                        JDs,
                        (obs.mean + obs.error * np.random.normal(size=obs.mean.shape))
                        * L**Ldim,
                        xc,
                        fitrange,
                    ),
                    xc=xc,
                    range=fitrange,
                )
            )
    return fitdata


def crossing(fit1, fit2):
    def y(x):
        return fit1(x) - fit2(x)

    bracket = [
        max(fit1.xc - fit1.range, fit2.xc - fit2.range),
        min(fit1.xc + fit1.range, fit2.xc + fit2.range),
    ]
    xcross = None
    try:
        xcross = spo.root_scalar(y, bracket=bracket).root
    except ValueError:
        pass

    return xcross


def calculate_crossings(fits):
    Ls = fits.keys()

    crossings = {}
    for L in Ls:
        if 2 * L in fits.keys():
            cross_samples = []
            for f1, f2 in zip(fits[L], fits[2 * L]):
                c = crossing(f1, f2)
                if c is not None:
                    cross_samples.append(c)

            crossings[2 * L] = (cross_samples[0], np.std(cross_samples[1:]))
    return crossings


def plot_binder(ax, mc, obsname, g0, colors, marker, crossdata=None, xlim=None, Ldim=0):
    handles = []

    cond = dict(max_photons=8)
    Ls = mc.get_parameter("Lx", unique=True)
    Ls = Ls[Ls <= 80]
    Ls = Ls[Ls >= 12]
    for i, L in enumerate(Ls):
        JDs = mc.get_parameter("JD", filter=dict(Lx=L, g0=g0, **cond))
        obs = mc.get_observable(obsname, filter=dict(Lx=L, g0=g0, **cond))
        ord = np.argsort(JDs)
        line = ax.errorbar(
            JDs[ord],
            obs.mean[ord] * L**Ldim,
            obs.error[ord] * L**Ldim,
            color=colors(i),
            marker=marker,
            ls="",
            label=f"$L={L}$",
            zorder=i,
        )
        if crossdata:
            pd = crossdata[g0][0][L][0].plot_data()
            ax.plot(pd[0], pd[1], "-", zorder=i, color=colors(i))

        handles.append(line)
    ax.set_xlim(xlim)
    ax.set_ylabel(obsname)
    # ax.legend(handles=handles, ncol=2)


def fig_binder(obsname, Ldim=0):
    crossdata = {}
    for obs, ldim in [(obsname, 0), ("MagChi", 1)]:
        crossdata[obs] = {}
        for g0 in [0, 1.5]:
            fd = calculate_fits(mc, obs, dict(g0=g0, max_photons=8), Ldim=ldim)
            crossdata[obs][g0] = (fd, calculate_crossings(fd))

    fig, axs = plt.subplots(1, 1, figsize=(3.375, 2.2), sharex=True, sharey=True)
    axs = [axs]
    xlim = [1.639, 1.647]
    axs[0].axvline(JDc, color="black", lw=0.5, zorder=-10)
    plot_binder(
        axs[0],
        mc,
        obsname,
        g0=0,
        marker=markers[0],
        colors=colorsets[0],
        crossdata=crossdata[obsname],
        xlim=xlim,
        Ldim=Ldim,
    )
    plot_binder(
        axs[0],
        mc,
        obsname,
        g0=1.5,
        marker=markers[1],
        colors=colorsets[1],
        crossdata=crossdata[obsname],
        xlim=xlim,
        Ldim=Ldim,
    )

    for ax in axs:
        ax.set_xlim(1.638, 1.647)
        ax.set_ylim([0.41, 0.47])
        ax.set_ylabel("$Q$")
    ax.set_xlabel("$J_\\mathrm{D}/J$")

    ax.text(0.63, 0.92, "$J_D^c/J = 1.6433(6)$", transform=ax.transAxes)
    ax.text(1.644, 0.456, f"$L={Ls[0]}$", rotation=-2, fontsize=8)
    ax.text(1.6435, 0.423, f"$L={Ls[-1]}$", rotation=-42, fontsize=8)

    axins = fig.add_axes([0.31, 0.34, 0.28, 0.22])

    plot_crossings(axins, crossdata[obsname], style=dict(ls="-"))
    plot_crossings(axins, crossdata["MagChi"], style=dict(ls=(2, (4, 2))))

    lines = [
        Line2D([0], [0], color="black", label="$Q$", ls="-"),
        Line2D([0], [0], color="black", label="$L\\chi$", ls=(1.8, (4, 2))),
    ]
    axins.legend(handles=lines, fontsize=6)

    handles = [
        Line2D(
            [0],
            [0],
            color=colorsets[0](5),
            marker=markers[0],
            label=f"$\\lambda = {0}$",
            ls="",
        ),
        Line2D(
            [0],
            [0],
            color=colorsets[1](5),
            marker=markers[1],
            label=f"$\\lambda = {1.5/3**0.5:.3g}$",
            ls="",
        ),
    ]

    axins.axhline(JDc, color="black", lw=0.5, zorder=-10)
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7,
        bbox_to_anchor=(0.34, 0.8),
    )
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.18)

    plt.savefig("../plots/honeycomb_crossings.pdf")
    plt.show()


def plot_crossings(ax, crossdata, style={}):
    def fitfunc(L, a, b):
        return a * L ** (-1 / 0.71209) + b

    def fitfuncL2(L, a, b, c):
        return a * L ** (-1 / 0.71209) * (1 + c * L**-2) + b

    handles = []
    for i, (g0, (_, xcross)) in enumerate(crossdata.items()):
        c = np.array(list(xcross.values()))
        Ls = 1.0 * np.array(list(xcross.keys()))
        clr = colorsets[i](5)

        handles.append(
            ax.errorbar(
                1 / Ls,
                c[:, 0],
                c[:, 1],
                color=clr,
                marker=markers[i],
                label=f"$\\lambda = {g0/3**0.5:.3g}$",
                **style,
            )
        )

    ax.tick_params(labelsize=6)
    ax.set_xlim([-0.000, None])
    ax.set_ylabel("$J_\\mathrm{D}/J$", fontsize=6)
    ax.set_xlabel("$1/L$", fontsize=6)
    return handles


fig_binder("StagUCBinderRatio", Ldim=0)
