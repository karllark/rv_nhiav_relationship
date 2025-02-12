import glob
import argparse
import matplotlib.pyplot as plt

import numpy as np
from math import sqrt, cos, sin
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from scipy.linalg import eigh

import astropy.units as u
from astropy.table import QTable

from measure_extinction.extdata import ExtData


def plot_exts(exts, rvs, avs, nhav, nhav_range, ctype, cwave, psym, label, alpha=0.5):
    oexts = get_alav(exts, ctype, cwave)
    xvals = rvs[:, 0]
    xvals_unc = rvs[:, 1]
    yvals = oexts[:, 0]
    yvals_unc = oexts[:, 1]
    avfrac = avs[:, 1] / avs[:, 0]
    ccols = 255. * (nhav - nhav_range[0]) / (nhav_range[1] - nhav_range[0])
    colors = cm.Reds(ccols.astype(int))
    ax[i].scatter(
        rvs[:, 0],
        oexts[:, 0],
        #psym,
        #fillstyle="none",
        label=label,
        #alpha=alpha,
        color=colors,
        marker=psym[1],
        # cmap='hsv'
    )
    # ax[i].errorbar(
    #     rvs[:, 0],
    #     oexts[:, 0],
    #     xerr=xvals_unc,
    #     yerr=yvals_unc,
    #     fmt=psym,
    #     fillstyle="none",
    #     alpha=0.2,
    # )
    return (xvals, xvals_unc, yvals, yvals_unc, avfrac)


# from Dries' dust_fuse_h2 repository
def cov_ellipse(x, y, cov, num_sigma=1, **kwargs):
    """
    Create an ellipse at the coordinates (x,y), that represents the
    covariance. The style of the ellipse can be adjusted using the
    kwargs.

    Returns
    -------
    ellipse: matplotlib.patches.Ellipse
    """

    position = [x, y]

    if cov[0, 1] != 0:
        # length^2 and orientation of ellipse axes is determined by
        # eigenvalues and vectors, respectively. Eigh is more stable for
        # symmetric / hermitian matrices.
        values, vectors = eigh(cov)
        width, height = np.sqrt(np.abs(values)) * num_sigma * 2
    else:
        width = sqrt(cov[0, 0]) * 2
        height = sqrt(cov[1, 1]) * 2
        vectors = np.array([[1, 0], [0, 1]])

    # I ended up using a Polygon just like Karl's plotting code. The
    # ellipse is buggy when the difference in axes is extreme (1e22). I
    # think it is because even a slight rotation will make the ellipse
    # look extremely strechted, as the extremely long axis (~1e22)
    # rotates into the short coordinate (~1).

    # two vectors representing the axes of the ellipse
    vw = vectors[:, 0] * width / 2
    vh = vectors[:, 1] * height / 2

    # generate corners
    num_corners = 64
    angles = np.linspace(0, 2 * np.pi, num_corners, endpoint=False)
    corners = np.row_stack([position + vw * cos(a) + vh * sin(a) for a in angles])

    return Polygon(corners, **kwargs)


def draw_ellipses(ax, xs, ys, covs, num_sigma=1, sigmas=None, **kwargs):
    for k, (x, y, cov) in enumerate(zip(xs, ys, covs)):
        # if sigmas is not None:
        #     color = cm.viridis(sigmas[k] / 3.0)[0]
        # ax.add_patch(cov_ellipse(x, y, cov, num_sigma, color=color, **kwargs))
        ax.add_patch(cov_ellipse(x, y, cov, num_sigma, **kwargs))


def cut_by_ebv(exts, rvs, avs, names, ebv_cut):
    """
    Remove all sightlines below the ebv cut
    """
    gvals, = np.where((avs[:, 0] / rvs[:, 0]) > ebv_cut)
    return np.array(exts)[gvals], rvs[gvals, :], avs[gvals, :], list(np.array(names)[gvals])


def cut_dups(names, exts, rvs, avs, dupnames):
    """
    Remove all that are in names
    """
    gvals = []
    for k, cname in enumerate(names):
        if cname not in dupnames:
            gvals.append(k)
    gvals = np.array(gvals)

    return np.array(exts)[gvals], rvs[gvals, :], avs[gvals, :]


def get_alav(exts, src, wave):
    """
    Get the A(lambda)/A(V) values for a particular wavelength for the sample
    """
    n_exts = len(exts)
    oext = np.full((n_exts, 2), np.nan)
    for i, iext in enumerate(exts):
        if src in iext.waves.keys():
            sindxs = np.argsort(np.absolute(iext.waves[src] - wave))
            if (iext.npts[src][sindxs[0]] > 0) and (iext.exts[src][sindxs[0]] > 0):
                oext[i, 0] = iext.exts[src][sindxs[0]]
                oext[i, 1] = iext.uncs[src][sindxs[0]]
            else:
                oext[i, 0] = np.nan
                oext[i, 1] = np.nan

    return oext


def get_irvs(rvs):
    """
    Compute 1/rvs values (including uncs) from rvs vals
    1/rvs have 1/3.1 subtracted for the fitting
    """
    irvs = np.zeros(rvs.shape)
    irvs[:, 0] = 1 / rvs[:, 0]
    irvs[:, 1] = irvs[:, 0] * (rvs[:, 1] / rvs[:, 0])
    irvs[:, 0] -= 1 / 3.1
    return irvs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # read in all the extinction curves
    files_gor09 = glob.glob("data/gor09*.fits")
    exts_gor09 = [ExtData(cfile) for cfile in files_gor09]
    psym_gor09 = "bs"

    files_gor24 = glob.glob("data/gor24*.fits")
    exts_gor24 = [ExtData(cfile) for cfile in files_gor24]
    psym_gor24 = "go"

    # get the ensemble parameters - NHI, etc.
    ensemble_gor09 = QTable.read("/home/kgordon/Python/extinction_ensemble_props/extinction_ensemble_props/data/GCC09_ensemble_params.dat", format="ipac")

    # get R(V) values
    n_gor09 = len(files_gor09)
    names_gor09 = []
    rvs_gor09 = np.zeros((n_gor09, 2))
    avs_gor09 = np.zeros((n_gor09, 2))
    nhs_gor09 = np.zeros((n_gor09, 2))
    for i, iext in enumerate(exts_gor09):
        tname = files_gor09[i].split("_")[1].lower()
        names_gor09.append(tname)

        av = iext.columns["AV"]
        avs_gor09[i, 0] = av[0]
        avs_gor09[i, 1] = av[1]

        irv = iext.columns["RV"]
        rvs_gor09[i, 0] = irv[0]
        rvs_gor09[i, 1] = irv[1]

        iext.trans_elv_alav()

        # get NH
        tname2 = tname.upper()
        gvals = tname.upper() == ensemble_gor09["Name"]
        nhs_gor09[i, 0] = (ensemble_gor09["NH"][gvals])[0]
        nhs_gor09[i, 1] = (ensemble_gor09["NH_unc"][gvals])[0]

    n_gor24 = len(files_gor24)
    names_gor24 = []
    rvs_gor24 = np.zeros((n_gor24, 2))
    avs_gor24 = np.zeros((n_gor24, 2))
    nhs_gor24 = np.zeros((n_gor24, 2))
    for i, iext in enumerate(exts_gor24):
        tname = files_gor24[i].split("_")[1].lower()
        names_gor24.append(tname)

        av = iext.columns["AV"]
        avs_gor24[i, 0] = av[0]
        avs_gor24[i, 1] = av[1]

        irv = iext.columns["RV"]
        rvs_gor24[i, 0] = irv[0]
        rvs_gor24[i, 1] = irv[1]

        nh = iext.columns["LOGHI"]
        nhs_gor24[i, 0] = 10**nh[0]
        nhs_gor24[i, 1] = 0.0

        iext.trans_elv_alav()

    # covert to 1/rv
    rvs_gor09 = get_irvs(rvs_gor09)
    rvs_gor24 = get_irvs(rvs_gor24)

    # compute nhi/av
    nhav_gor09 = nhs_gor09[:, 0] / avs_gor09[:, 0]
    nhav_gor24 = nhs_gor24[:, 0] / avs_gor24[:, 0]
    minnhav = np.min([np.min(nhav_gor09), np.min(nhav_gor24)])
    maxnhav = np.max([np.max(nhav_gor09), np.max(nhav_gor24)])
    print("nhav range", minnhav, maxnhav)

    labx = "$1/R(V)$ - 1/3.1"
    xrange = np.array([1.0 / 6.5, 1.0 / 2.0]) - 1 / 3.1
    laby = r"$A(\lambda)/A(V)$"

    fontsize = 12
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    do_2dcorr = True

    nsteps = 1000

    leg_fontsize = 0.8 * fontsize
 
    fig, fax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 12),
        sharex=True,  # constrained_layout=True
    )

    repwaves = {
        "IUE0": 0.135 * u.micron,
        "IUE2": 0.2175 * u.micron,
        "IUE3": 0.3 * u.micron,
        "BAND1": 1.1 * u.micron,
    }
    ax = fax.flatten()

    for i, rname in enumerate(repwaves.keys()):
        xvals = None
        yvals = None
        xvals_unc = None
        yvals_unc = None
        avfrac = None

        if "FUSE" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor09,
                rvs_gor09,
                avs_gor09,
                nhav_gor09,
                [minnhav, maxnhav],
                "FUSE",
                repwaves[rname],
                psym_gor09,
                "GCC09",
            )

        elif "IUE" in rname:
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor09,
                rvs_gor09,
                avs_gor09,
                nhav_gor09,
                [minnhav, maxnhav],
                "IUE",
                repwaves[rname],
                psym_gor09,
                "GCC09",
            )
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor24,
                rvs_gor24,
                avs_gor24,
                nhav_gor24,
                [minnhav, maxnhav],
                "IUE",
                repwaves[rname],
                psym_gor24,
                "G24",
            )

        elif "BAND" in rname:

            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor09,
                rvs_gor09,
                avs_gor09,
                nhav_gor09,
                [minnhav, maxnhav],
                "BAND",
                repwaves[rname],
                psym_gor09,
                "GCC09",
            )
            xvals, xvals_unc, yvals, yvals_unc, avfrac = plot_exts(
                exts_gor24,
                rvs_gor24,
                avs_gor24,
                nhav_gor24,
                [minnhav, maxnhav],
                "BAND",
                repwaves[rname],
                psym_gor24,
                "G24",
            )

        leg = ax[i].legend(ncol=2, fontsize=leg_fontsize)
        leg.set_title(f"{repwaves[rname]}", prop={"size": leg_fontsize})


    ax[0].set_xlim(xrange)

    # for 2nd x-axis with R(V) values
    axis_rvs = np.array([2.3, 2.5, 3.0, 4.0, 5.0, 6.0])
    new_ticks = 1 / axis_rvs - 1 / 3.1
    new_ticks_labels = ["%.1f" % z for z in axis_rvs]

    for i in range(2):
        fax[1, i].set_xlabel(labx)

        # add 2nd x-axis with R(V) values
        tax = fax[0, i].twiny()
        tax.set_xlim(fax[0, i].get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$R(V)$")

    for i in range(2):
        fax[i, 0].set_ylabel(laby)

    fig.tight_layout()

    fname = "rv_nhiav_rep_waves"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
