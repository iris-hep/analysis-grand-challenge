import uproot


def save_histograms(hist_dict, filename, add_offset=False):
    with uproot.recreate(filename) as f:
        # save all available histograms to disk
        for channel, histogram in hist_dict.items():
            # optionally add minimal offset to avoid completely empty bins
            # (useful for the ML validation variables that would need binning adjustment
            # to avoid those)
            if add_offset:
                histogram += 1e-6

            for sample in histogram.axes[1]:
                for variation in histogram[:, sample, :].axes[1]:
                    variation_string = "" if variation == "nominal" else f"_{variation}"
                    current_1d_hist = histogram[:, sample, variation]

                    if sum(current_1d_hist.values()) != 0:
                        # only save histograms containing events
                        # many combinations are not used (e.g. ME var for W+jets)
                        f[f"{channel}_{sample}{variation_string}"] = current_1d_hist

            # add pseudodata histogram if all inputs to it are available
            if (
                sum(histogram[:, "ttbar", "ME_var"].values()) != 0
                and sum(histogram[:, "ttbar", "PS_var"].values()) != 0
                and sum(histogram[:, "wjets", "nominal"].values()) != 0
            ):
                f[f"{channel}_pseudodata"] = (
                    histogram[:, "ttbar", "ME_var"] + histogram[:, "ttbar", "PS_var"]
                ) / 2 + histogram[:, "wjets", "nominal"]
