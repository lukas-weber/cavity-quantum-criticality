import h5py
import matplotlib.pyplot as plt
import numpy as np
import mcextract
import scipy.ndimage as spndi


def plot_sign_free(fig, ax):
    with h5py.File("../data/sign/sign_free.h5", "r") as f:
        omegas = f["omegas"][...]
        gs = f["gs"][...]

        results = {}
        for result in f["results"]:
            nphot = int(result.split("=")[-1])
            results[nphot] = f["results"][result][...].T

    momegas, mgs = np.meshgrid(omegas, gs)

    cm = plt.get_cmap("PuBuGn")

    colors = [cm(0.2), cm(0.5), cm(0.999)]

    nphots = results.keys()

    for c, nphot in zip(colors, nphots):
        # gaussian filter to allow finding contours smoothly. sigma should be smaller than curvature radius of the contour.
        data = spndi.gaussian_filter(results[nphot], sigma=4)
        ax.contourf(mgs, momegas, data, [0.5, 1.1], antialiased=True, colors=[c])

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=clr, ec="black", lw=0.5) for clr in colors]
    ax.legend(
        reversed(proxy),
        [f"{nphot}" for nphot in reversed(nphots)],
        title="$n_\\mathrm{ph}^\\mathrm{max}=$",
        fontsize=7,
        title_fontsize=7,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.43),
    )
    ax.text(
        0.5,
        0.9,
        "$\\langle n|\\hat{\\mathcal{J}}_{ij}|m\\rangle \\ge 0$",
        ha="center",
        va="center",
        fontsize=7,
    )

    ax.text(0.1, 1.4, "(a)", color="white", ha="center", va="center")
    ax.set_ylabel("$\\Omega/U$")
    ax.set_xlabel("$\\lambda/\\sqrt{N}$")


def plot_sign_practice(fig, ax):
    mc = mcextract.MCArchive("../data/sign/mc.json")
    max_photons = 8
    coupling_fac = 3**-0.5
    obsname = "Sign"
    U = mc.get_parameter("U", unique=True, filter=dict(max_photons=max_photons))
    gs = mc.get_parameter("g", unique=True, filter=dict(max_photons=max_photons))
    omegas = mc.get_parameter(
        "omega", unique=True, filter=dict(max_photons=max_photons)
    )

    L = mc.get_parameter("Lx", unique=True)[0]

    data = np.zeros([len(gs), len(omegas)])

    for i, g in enumerate(gs):
        data[
            i,
        ] = mc.get_observable(obsname, filter=dict(g=g, max_photons=max_photons)).mean

    momegas, mgs = np.meshgrid(omegas, gs)

    def close_to_resonance(omega, tolerance):
        for n in range(1, round(1 / tolerance)):
            if abs(omega - 1 / n) < tolerance:
                return True
        return False

    pclr = ax.pcolormesh(
        mgs * coupling_fac,
        momegas / U,
        data,
        vmax=1,
        vmin=0,
        cmap="Greens",
        rasterized=True,
    )

    ax.text(0.14, 1.4, "(b)", color="white", ha="center", va="center")
    ax.text(
        0.5,
        0.9,
        "$\\langle\\mathrm{sign}\\rangle$",
        ha="center",
        va="center",
        fontsize=7,
    )

    ax.text(
        0.5,
        0.43 * 1.5,
        f"$n_\\mathrm{{ph}}^\\mathrm{{max}} = 8$\n$L={L}$",
        ha="center",
        va="center",
        fontsize=7,
    )
    ax.set_xlabel("$\\lambda/\\sqrt{N}$")
    fig.colorbar(pclr, label="$\\langle \\mathrm{sign}\\rangle$", ax=ax)


fig, axs = plt.subplots(
    1, 2, sharex=True, sharey=True, gridspec_kw={"width_ratios": [0.85, 1]}
)

plot_sign_free(fig, axs[0])
plot_sign_practice(fig, axs[1])

for ax in axs:
    ax.set_xlim(0.01, 1)
    ax.set_ylim(0, 1.5)
plt.tight_layout(pad=0.1)
plt.savefig("../plots/sign.pdf", dpi=400)
plt.show()
