labels_ = {
    "StagXStagYMagChi": "J\\chi(\\pi,\\pi)",
    "StagXStagYMag2": "S(\\pi,\\pi)/N",
    "StagXStagUCMagChi": "J\\chi(\\pi,\\pi)",  # because of unit cell sign pattern
    "StagXStagUCMag2": "S(\\pi,\\pi)",
    "StagUCMag2": "S",
    "Mag2": "m^2",
    "BinderRatio": "Q",
    "MagChi": "\\chi",
    "Energy": "E",
    "PhotonNum": "n_{\\mathrm{ph}}",
    "PhotonNumVar": "\\langle n_{\\mathrm{ph}}^2\\rangle - \\langle n_{\\mathrm{ph}}\\rangle^2",
    "SpecificHeat": "C",
    "Sign": "\\mathrm{sign}",
}


def label(obsname):
    return labels_.get(obsname, f"\\mathrm{{{obsname}}}")


def Ldim(obsname):
    if obsname.endswith("Mag2"):
        return 2
    return 0
