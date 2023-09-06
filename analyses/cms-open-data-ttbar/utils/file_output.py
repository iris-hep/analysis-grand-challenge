import uproot


def save_histograms(hist_dict, fileset, filename, channel_names, add_offset=False):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    with uproot.recreate(filename) as f:
        out_dict = {}
        for channel in channel_names:
            current_hist = hist_dict[channel]

            # optionally add minimal offset to avoid completely empty bins
            # (useful for the ML validation variables that would need binning adjustment to avoid those)
            if add_offset:
                current_hist += 1e-6

            out_dict[f"{channel}_pseudodata"] = ((current_hist[:, "ttbar", "ME_var"] + current_hist[:, "ttbar", "PS_var"]) / 2 
                                                 + current_hist[:, "wjets", "nominal"])

            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                out_dict[f"{channel}_{sample_name}"] = current_hist[:, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        out_dict[f"{channel}_{sample_name}_{variation_name}"] = current_hist[:, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    out_dict[f"{channel}_{sample_name}_{variation_name}"] = current_hist[:, sample_name, variation_name]

            # ttbar modeling
            ttbar_variations = ["ME_var", "PS_var", "scaleup", "scaledown"]
            for var in ttbar_variations:
                out_dict[f"{channel}_ttbar_{var}"] = current_hist[:, "ttbar", var]

            # W+jets scale
            wjets_variations = ["scale_var_down", "scale_var_up"]
            for var in wjets_variations:
                out_dict[f"{channel}_wjets_{var}"] = current_hist[:, "wjets", var]

        # write to file
        for key in out_dict.keys():
            f[key] = out_dict[key]
