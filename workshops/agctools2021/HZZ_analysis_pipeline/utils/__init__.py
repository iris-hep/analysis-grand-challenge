import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


def clean_up():
    # clean up files that may be left over from previous running (not needed, just to
    # simplify debugging)
    for path in ["histograms/", "figures/"]:
        if os.path.exists(path):
            shutil.rmtree(path)


def set_logging():
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s")
    logging.getLogger("cabinetry").setLevel(logging.INFO)


def plot_errorband(bin_edge_low, bin_edge_high, num_bins, histograms):
    # bin parameters
    step_size = (bin_edge_high - bin_edge_low) / num_bins
    bin_centers = histograms["data"].axes[0].centers

    # calculate background statistical uncertainty
    mc_histogram_sum = histograms["MC"][:, :, "nominal"].project("mllll").values()
    mc_err = np.sqrt(histograms["MC"][:, :, "nominal"].project("mllll").variances())

    # plot background statistical uncertainty
    plt.bar(
        bin_centers,  # x
        2 * mc_err,  # heights
        alpha=0.5,  # half transparency
        bottom=mc_histogram_sum - mc_err,
        color="none",
        hatch="////",
        width=step_size,
        label="Stat. Unc.",
    )

    # tune plot appearance
    main_axes = plt.gca()
    main_axes.set_xlim(left=bin_edge_low, right=bin_edge_high)
    main_axes.set_ylim(bottom=0, top=np.amax(histograms["data"].values()) * 1.6)
    main_axes.set_ylabel("Events / " + str(step_size) + " GeV")
    main_axes.legend(frameon=False)


def save_figure(figname: str):
    """Takes the current figure and saves it in .pdf and .png formats under figures/."""
    fig = plt.gcf()
    fig.set_facecolor("white")

    if not os.path.exists("figures/"):
        os.mkdir("figures/")

    for filetype in ["pdf", "png"]:
        fig.savefig(f"figures/{figname}.{filetype}")
