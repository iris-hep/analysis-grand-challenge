import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def set_style():
    mpl.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "222222"
    plt.rcParams["axes.labelcolor"] = "222222"
    plt.rcParams["xtick.color"] = "222222"
    plt.rcParams["ytick.color"] = "222222"
    plt.rcParams["font.size"] = 10
    plt.rcParams["text.color"] = "222222"


def plot_data_mc(model_prediction_prefit, model_prediction_postfit, data, config):
    """
    Display data-MC comparison plots. Modeled using cabinetry's visualize.data_mc but displays in a less cumbersome way (and shows prefit and postfit plots alongside one another.

    Args:
        model_prediction_prefit (model_utils.ModelPrediction): prefit model prediction to show
        model_prediction_postfit (model_utils.ModelPrediction): postfit model prediction to show
        data (List[float]): data to include in visualization
        config (Optional[Dict[str, Any]], optional): cabinetry configuration needed for binning and axis labels

    Returns:
        figs: list of matplotlib Figures containing prefit and postfit plots
    """

    n_bins_total = sum(model_prediction_prefit.model.config.channel_nbins.values())
    if len(data) != n_bins_total:
        data = data[:n_bins_total]

    data_yields = [
        data[model_prediction_prefit.model.config.channel_slices[ch]]
        for ch in model_prediction_prefit.model.config.channels
    ]

    channels = model_prediction_prefit.model.config.channels
    channel_indices = [
        model_prediction_prefit.model.config.channels.index(ch) for ch in channels
    ]

    figs = []
    for i_chan, channel_name in zip(channel_indices, channels):
        histogram_dict_list_prefit = []  # one dict per region/channel
        histogram_dict_list_postfit = []  # one dict per region/channel

        regions = [reg for reg in config["Regions"] if reg["Name"] == channel_name]
        region_dict = regions[0]
        bin_edges = np.asarray(region_dict["Binning"])
        variable = region_dict.get("Variable", "bin")

        for i_sam, sample_name in enumerate(
            model_prediction_prefit.model.config.samples
        ):
            histogram_dict_list_prefit.append(
                {
                    "label": sample_name,
                    "isData": False,
                    "yields": model_prediction_prefit.model_yields[i_chan][i_sam],
                    "variable": variable,
                }
            )
            histogram_dict_list_postfit.append(
                {
                    "label": sample_name,
                    "isData": False,
                    "yields": model_prediction_postfit.model_yields[i_chan][i_sam],
                    "variable": variable,
                }
            )

        # add data sample
        histogram_dict_list_prefit.append(
            {
                "label": "Data",
                "isData": True,
                "yields": data_yields[i_chan],
                "variable": variable,
            }
        )
        histogram_dict_list_postfit.append(
            {
                "label": "Data",
                "isData": True,
                "yields": data_yields[i_chan],
                "variable": variable,
            }
        )

        label_prefit = f"{channel_name}\n{model_prediction_prefit.label}"
        label_postfit = f"{channel_name}\n{model_prediction_postfit.label}"

        mc_histograms_yields_prefit = []
        mc_histograms_yields_postfit = []
        mc_labels = []
        for i in range(len(histogram_dict_list_prefit)):
            if histogram_dict_list_prefit[i]["isData"]:
                data_histogram_yields = histogram_dict_list_prefit[i]["yields"]
                data_histogram_stdev = np.sqrt(data_histogram_yields)
                data_label = histogram_dict_list_prefit[i]["label"]
            else:
                mc_histograms_yields_prefit.append(
                    histogram_dict_list_prefit[i]["yields"]
                )
                mc_histograms_yields_postfit.append(
                    histogram_dict_list_postfit[i]["yields"]
                )
                mc_labels.append(histogram_dict_list_prefit[i]["label"])

        mpl.style.use("seaborn-v0_8-colorblind")

        fig = plt.figure(figsize=(12, 6), layout="constrained")
        gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])

        # increase font sizes
        for item in (
            [
                ax1.yaxis.label,
                ax2.xaxis.label,
                ax2.yaxis.label,
                ax3.yaxis.label,
                ax4.xaxis.label,
                ax4.yaxis.label,
            ]
            + ax1.get_yticklabels()
            + ax2.get_xticklabels()
            + ax2.get_yticklabels()
            + ax3.get_yticklabels()
            + ax4.get_xticklabels()
            + ax4.get_yticklabels()
        ):
            item.set_fontsize("large")

        # plot MC stacked together
        total_yield_prefit = np.zeros_like(mc_histograms_yields_prefit[0])
        total_yield_postfit = np.zeros_like(mc_histograms_yields_postfit[0])

        bin_right_edges = bin_edges[1:]
        bin_left_edges = bin_edges[:-1]
        bin_width = bin_right_edges - bin_left_edges
        bin_centers = 0.5 * (bin_left_edges + bin_right_edges)

        mc_containers_prefit = []
        mc_containers_postfit = []

        for i in range(len(mc_histograms_yields_prefit)):
            mc_container = ax1.bar(
                bin_centers,
                mc_histograms_yields_prefit[i],
                width=bin_width,
                bottom=total_yield_prefit,
            )
            mc_containers_prefit.append(mc_container)
            mc_container = ax3.bar(
                bin_centers,
                mc_histograms_yields_postfit[i],
                width=bin_width,
                bottom=total_yield_postfit,
            )
            mc_containers_postfit.append(mc_container)

            # add a black line on top of each sample
            line_x = [y for y in bin_edges for _ in range(2)][1:-1]
            line_y = [
                y
                for y in (mc_histograms_yields_prefit[i] + total_yield_prefit)
                for _ in range(2)
            ]
            ax1.plot(line_x, line_y, "-", color="black", linewidth=0.5)
            line_y = [
                y
                for y in (mc_histograms_yields_postfit[i] + total_yield_postfit)
                for _ in range(2)
            ]
            ax3.plot(line_x, line_y, "-", color="black", linewidth=0.5)

            total_yield_prefit += mc_histograms_yields_prefit[i]
            total_yield_postfit += mc_histograms_yields_postfit[i]

        # add total MC uncertainty
        total_model_unc_prefit = np.asarray(
            model_prediction_prefit.total_stdev_model_bins[i_chan][-1]
        )
        mc_unc_container_prefit = ax1.bar(
            bin_centers,
            2 * total_model_unc_prefit,
            width=bin_width,
            bottom=total_yield_prefit - total_model_unc_prefit,
            fill=False,
            linewidth=0,
            edgecolor="gray",
            hatch=3 * "/",
        )
        total_model_unc_postfit = np.asarray(
            model_prediction_postfit.total_stdev_model_bins[i_chan][-1]
        )
        mc_unc_container_postfit = ax3.bar(
            bin_centers,
            2 * total_model_unc_postfit,
            width=bin_width,
            bottom=total_yield_postfit - total_model_unc_postfit,
            fill=False,
            linewidth=0,
            edgecolor="gray",
            hatch=3 * "/",
        )

        # plot data
        data_container_prefit = ax1.errorbar(
            bin_centers,
            data_histogram_yields,
            yerr=data_histogram_stdev,
            fmt="o",
            color="k",
        )
        data_container_postfit = ax3.errorbar(
            bin_centers,
            data_histogram_yields,
            yerr=data_histogram_stdev,
            fmt="o",
            color="k",
        )

        # ratio plots
        ax2.plot(
            [bin_left_edges[0], bin_right_edges[-1]],
            [1, 1],
            "--",
            color="black",
            linewidth=1,
        )  # reference line along y=1
        ax4.plot(
            [bin_left_edges[0], bin_right_edges[-1]],
            [1, 1],
            "--",
            color="black",
            linewidth=1,
        )  # reference line along y=1

        n_zero_pred_prefit = sum(
            total_yield_prefit == 0.0
        )  # number of bins with zero predicted yields
        if n_zero_pred_prefit > 0:
            logging.warning(
                f"(PREFIT) predicted yield is zero in {n_zero_pred_prefit} bin(s), excluded from ratio plot"
            )
        nonzero_model_yield_prefit = total_yield_prefit != 0.0
        if np.any(total_yield_prefit < 0.0):
            raise ValueError(
                f"(PREFIT) {label_prefit} total model yield has negative bin(s): {total_yield_prefit.tolist()}"
            )

        n_zero_pred_postfit = sum(
            total_yield_postfit == 0.0
        )  # number of bins with zero predicted yields
        if n_zero_pred_postfit > 0:
            logging.warning(
                f"(POSTFIT) predicted yield is zero in {n_zero_pred_postfit} bin(s), excluded from ratio plot"
            )
        nonzero_model_yield_postfit = total_yield_postfit != 0.0
        if np.any(total_yield_postfit < 0.0):
            raise ValueError(
                f"(POSTFIT) {label_postfit} total model yield has negative bin(s): {total_yield_postfit.tolist()}"
            )

        # add uncertainty band around y=1
        rel_mc_unc_prefit = total_model_unc_prefit / total_yield_prefit
        rel_mc_unc_postfit = total_model_unc_postfit / total_yield_postfit
        # do not show band in bins where total model yield is 0
        ax2.bar(
            bin_centers[nonzero_model_yield_prefit],
            2 * rel_mc_unc_prefit[nonzero_model_yield_prefit],
            width=bin_width[nonzero_model_yield_prefit],
            bottom=1.0 - rel_mc_unc_prefit[nonzero_model_yield_prefit],
            fill=False,
            linewidth=0,
            edgecolor="gray",
            hatch=3 * "/",
        )
        ax4.bar(
            bin_centers[nonzero_model_yield_postfit],
            2 * rel_mc_unc_postfit[nonzero_model_yield_postfit],
            width=bin_width[nonzero_model_yield_postfit],
            bottom=1.0 - rel_mc_unc_postfit[nonzero_model_yield_postfit],
            fill=False,
            linewidth=0,
            edgecolor="gray",
            hatch=3 * "/",
        )

        # data in ratio plot
        data_model_ratio_prefit = data_histogram_yields / total_yield_prefit
        data_model_ratio_unc_prefit = data_histogram_stdev / total_yield_prefit
        data_model_ratio_postfit = data_histogram_yields / total_yield_postfit
        data_model_ratio_unc_postfit = data_histogram_stdev / total_yield_postfit
        # mask data in bins where total model yield is 0
        ax2.errorbar(
            bin_centers[nonzero_model_yield_prefit],
            data_model_ratio_prefit[nonzero_model_yield_prefit],
            yerr=data_model_ratio_unc_prefit[nonzero_model_yield_prefit],
            fmt="o",
            color="k",
        )
        ax4.errorbar(
            bin_centers[nonzero_model_yield_postfit],
            data_model_ratio_postfit[nonzero_model_yield_postfit],
            yerr=data_model_ratio_unc_postfit[nonzero_model_yield_postfit],
            fmt="o",
            color="k",
        )

        # get the highest single bin yield, from the sum of MC or data
        y_max = max(
            max(np.amax(total_yield_prefit), np.amax(data_histogram_yields)),
            max(np.amax(total_yield_postfit), np.amax(data_histogram_yields)),
        )

        ax1.set_ylim([0, y_max * 1.5])  # 50% headroom
        ax3.set_ylim([0, y_max * 1.5])  # 50% headroom

        # figure label (region name)
        at = mpl.offsetbox.AnchoredText(
            label_prefit,
            loc="upper left",
            frameon=False,
            prop={"fontsize": "large", "linespacing": 1.5},
        )
        ax1.add_artist(at)
        at = mpl.offsetbox.AnchoredText(
            label_postfit,
            loc="upper left",
            frameon=False,
            prop={"fontsize": "large", "linespacing": 1.5},
        )
        ax3.add_artist(at)

        # MC contributions in inverse order, such that first legend entry corresponds to
        # the last (highest) contribution to the stack
        all_containers = mc_containers_prefit[::-1] + [
            mc_unc_container_prefit,
            data_container_prefit,
        ]
        all_labels = mc_labels[::-1] + ["Uncertainty", data_label]
        ax1.legend(
            all_containers,
            all_labels,
            frameon=False,
            fontsize="large",
            loc="upper right",
        )
        all_containers = mc_containers_postfit[::-1] + [
            mc_unc_container_postfit,
            data_container_postfit,
        ]
        ax3.legend(
            all_containers,
            all_labels,
            frameon=False,
            fontsize="large",
            loc="upper right",
        )

        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax1.set_ylabel("events")
        ax1.set_xticklabels([])
        ax1.set_xticklabels([], minor=True)
        ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
        ax1.tick_params(direction="in", top=True, right=True, which="both")

        ax3.set_xlim(bin_edges[0], bin_edges[-1])
        ax3.set_xticklabels([])
        ax3.set_xticklabels([], minor=True)
        ax3.set_yticklabels([])
        ax3.set_yticklabels([], minor=True)
        ax3.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
        ax3.tick_params(direction="in", top=True, right=True, which="both")

        ax2.set_xlim(bin_edges[0], bin_edges[-1])
        ax2.set_ylim([0.5, 1.5])
        ax2.set_xlabel(histogram_dict_list_prefit[0]["variable"])
        ax2.set_ylabel("data / model")
        ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
        ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
        ax2.tick_params(axis="both", which="major", pad=8)
        ax2.tick_params(direction="in", top=True, right=True, which="both")

        ax4.set_xlim(bin_edges[0], bin_edges[-1])
        ax4.set_ylim([0.5, 1.5])
        ax4.set_xlabel(histogram_dict_list_postfit[0]["variable"])
        ax4.set_yticklabels([])
        ax4.set_yticklabels([], minor=True)
        ax4.tick_params(axis="both", which="major", pad=8)
        ax4.tick_params(direction="in", top=True, right=True, which="both")

        fig.show()

        figs.append({"figure": fig, "region": channel_name})

    return figs

def plot_training_variables(all_correct, some_correct, none_correct):
    
    import hist
    from .config import config as config
    config = config["ml"]
    
    legend_list = ["All Matches Correct", "Some Matches Correct", "No Matches Correct"]
    
    n_features = all_correct.shape[1]
        
    # plot features on grid
    fig, axs = plt.subplots(10,2,figsize=(14,40))
    for i in range(n_features):
        if i<10: 
            column=0
            row=i
        else: 
            column=1
            row=i-10
            
        # define histogram
        h = hist.Hist(
            hist.axis.Regular(50, config["BIN_LOW"][i], config["BIN_HIGH"][i], name='observable', label=config["FEATURE_DESCRIPTIONS"][i], flow=False),
            hist.axis.StrCategory(legend_list, name="truthlabel", label="Truth Label"),
        )
        
        # fill histogram
        h.fill(observable = all_correct[:,i], truthlabel="All Matches Correct")
        h.fill(observable = some_correct[:,i], truthlabel="Some Matches Correct")
        h.fill(observable = none_correct[:,i], truthlabel="No Matches Correct")
        
        h.plot(density=True, ax=axs[row,column])
        axs[row, column].legend(legend_list, frameon=False)
    
    fig.show()
