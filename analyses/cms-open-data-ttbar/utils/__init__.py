import asyncio
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot
from servicex import ServiceXDataset
import numpy as np
import awkward as ak

def get_client(af="coffea_casa"):
    if af == "coffea_casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from lpcdaskgateway import LPCGateway

        gateway = LPCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: {str(cluster.dashboard_link)}")

        client = cluster.get_client()

    elif af == "local":
        from dask.distributed import Client

        client = Client()

    else:
        raise NotImplementedError(f"unknown analysis facility: {af}")

    return client


def set_style():
    mpl.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "222222"
    plt.rcParams["axes.labelcolor"] = "222222"
    plt.rcParams["xtick.color"] = "222222"
    plt.rcParams["ytick.color"] = "222222"
    plt.rcParams["font.size"] = 10
    plt.rcParams['text.color'] = "222222"


def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name=""):
    # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
    # for reference
    # x-secs are in pb
    xsec_info = {
        "ttbar": 396.87 + 332.97, # nonallhad + allhad, keep same x-sec for all
        "single_top_s_chan": 2.0268 + 1.2676,
        "single_top_t_chan": (36.993 + 22.175)/0.252,  # scale from lepton filter to inclusive
        "single_top_tW": 37.936 + 37.906,
        "wjets": 61457 * 0.252,  # e/mu+nu final states
        "data": None
    }

    # list of files
    with open("nanoaod_inputs.json") as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        if process == "data":
            continue  # skip data

        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[:n_files_max_per_sample]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://red-xcache1.unl.edu") for f in file_paths]
            if af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/", "/data/alheld/") for f in file_paths]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


def save_histograms(all_histograms, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    all_histograms += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    pseudo_data = (all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 2  + all_histograms[:, :, "wjets", "nominal"]

    with uproot.recreate(filename) as f:
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), region]
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

            # ttbar modeling
            f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "ME_var"]
            f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "PS_var"]

            f[f"{region}_ttbar_scaledown"] = all_histograms[120j :: hist.rebin(2), region, "ttbar", "scaledown"]
            f[f"{region}_ttbar_scaleup"] = all_histograms[120j :: hist.rebin(2), region, "ttbar", "scaleup"]

            # W+jets scale
            f[f"{region}_wjets_scale_var_down"] = all_histograms[120j :: hist.rebin(2), region, "wjets", "scale_var_down"]
            f[f"{region}_wjets_scale_var_up"] = all_histograms[120j :: hist.rebin(2), region, "wjets", "scale_var_up"]
            
            
def save_ml_histograms(hist_dict, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    features = ["deltar_leptontoplep","deltar_w1w2","deltar_w1tophad","deltar_w2tophad","mass_leptontoplep","mass_w1w2",
                "mass_w1w2tophad","pt_w1w2tophad","pt_w1","pt_w2","pt_tophad","pt_toplep",
                "btag_w1","btag_w2","btag_tophad","btag_toplep","qgl_w1","qgl_w2","qgl_tophad","qgl_toplep"]
    for feature in features:
        hist_dict[f"hist_{feature}"] += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    with uproot.recreate(filename) as f:
        for feature in features:
            current_hist = hist_dict[f"hist_{feature}"]
            f[f"{feature}_pseudodata"] = (current_hist[:, "ttbar", "ME_var"] + current_hist[:, "ttbar", "PS_var"]) / 2  + current_hist[:, "wjets", "nominal"]
            
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{feature}_{sample_name}"] = current_hist[:, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{feature}_{sample_name}_{variation_name}"] = current_hist[:, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{feature}_{sample_name}_{variation_name}"] = current_hist[:, sample_name, variation_name]

            # ttbar modeling
            f[f"{feature}_ttbar_ME_var"] = current_hist[:, "ttbar", "ME_var"]
            f[f"{feature}_ttbar_PS_var"] = current_hist[:, "ttbar", "PS_var"]

            f[f"{feature}_ttbar_scaledown"] = current_hist[:, "ttbar", "scaledown"]
            f[f"{feature}_ttbar_scaleup"] = current_hist[:, "ttbar", "scaleup"]

            # W+jets scale
            f[f"{feature}_wjets_scale_var_down"] = current_hist[:, "wjets", "scale_var_down"]
            f[f"{feature}_wjets_scale_var_up"] = current_hist[:, "wjets", "scale_var_up"]


class ServiceXDatasetGroup():
    def __init__(self, fileset, backend_name="uproot", ignore_cache=False):
        self.fileset = fileset

        # create list of files (& associated processes)
        filelist = []
        for i, process in enumerate(fileset):
            filelist += [[filename, process] for filename in fileset[process]["files"]]

        filelist = np.array(filelist)
        self.filelist = filelist
        self.ds = ServiceXDataset(filelist[:,0].tolist(), backend_name=backend_name, ignore_cache=ignore_cache)

    def get_data_rootfiles_uri(self, query, as_signed_url=True, title="Untitled"):

        all_files = np.array(self.ds.get_data_rootfiles_uri(query, as_signed_url=as_signed_url, title=title))
        parent_file_urls = np.array([f.file for f in all_files])

        # order is not retained after transform, so we can match files to their parent files using the filename
        # (replacing / with : to mitigate servicex filename convention )
        parent_key = np.array([np.where(parent_file_urls==self.filelist[i][0].replace("/",":"))[0][0]
                               for i in range(len(self.filelist))])

        files_per_process = {}
        for i, process in enumerate(self.fileset):
            # update files for each process
            files_per_process.update({process: all_files[parent_key[self.filelist[:,1]==process]]})

        return files_per_process

    
def get_permutations_dict(MAX_N_JETS, include_labels=False, include_eval_mat=False):
    '''
    Get dictionary which lists the different permutations considered for each event, depending on number of jets in an event.
    
    Args:
        MAX_N_JETS: maximum number of jets to consider for permutations (ordered by pT)
        include_labels: whether to include another dictionary with associated labels for each permutation 
                        (24=W jet, 6=top_hadron jet, -6=top_lepton jet)
        include_eval_mat: whether to include a dictionary of matrices which calculates the associated jet score between predicted and truth labels
    
    Returns:
        permutations_dict: dictionary containing lists of permutation indices for each number of jets up to MAX_N_JETS
        labels_dict (optional): dictionary containing associated labels of permutations listed in permutations_dict
        evaluation_matrices (optional): dictionary containing associated evaluation matrices to get the associated jet score using the predicted and truth labels
    '''
    
    # calculate the dictionary of permutations for each number of jets
    permutations_dict = {}
    for n in range(4,MAX_N_JETS+1):
        test = ak.Array(range(n))
        unzipped = ak.unzip(ak.argcartesian([test]*4,axis=0))

        combos = ak.combinations(ak.Array(range(4)), 2, axis=0)
        different = unzipped[combos[0]["0"]]!=unzipped[combos[0]["1"]]
        for i in range(1,len(combos)):
            different = different & (unzipped[combos[i]["0"]]!=unzipped[combos[i]["1"]])

        permutations = ak.zip([test[unzipped[i][different]] for i in range(len(unzipped))],
                              depth_limit=1).tolist()


        permutations = ak.concatenate([test[unzipped[i][different]][..., np.newaxis] 
                                       for i in range(len(unzipped))], 
                                      axis=1).to_list()

        permutations_dict[n] = permutations

    # for each permutation, calculate the corresponding label
    labels_dict = {}
    for n in range(4,MAX_N_JETS+1):

        current_labels = []
        for inds in permutations_dict[n]:

            inds = np.array(inds)
            current_label = 100*np.ones(n)
            current_label[inds[:2]] = 24
            current_label[inds[2]] = 6
            current_label[inds[3]] = -6
            current_labels.append(current_label.tolist())

        labels_dict[n] = current_labels

    # get rid of duplicates since we consider W jets to be exchangeable
    # (halves the number of permutations we consider)
    for n in range(4,MAX_N_JETS+1):
        res = []
        for idx, val in enumerate(labels_dict[n]):
            if val in labels_dict[n][:idx]:
                res.append(idx)
        labels_dict[n] = np.array(labels_dict[n])[res].tolist()
        permutations_dict[n] = np.array(permutations_dict[n])[res].tolist()
        print("number of permutations for n=",n,": ", len(permutations_dict[n]))
        
    if include_labels and not include_eval_mat:
        return permutations_dict, labels_dict
    
    elif include_labels and include_eval_mat:
        # these matrices tell you the overlap between the predicted label (rows) and truth label (columns)
        # the "score" in each matrix entry is the number of jets which are assigned correctly        
        evaluation_matrices = {}  # overall event score

        for n in range(4, MAX_N_JETS+1):
            evaluation_matrix = np.zeros((len(permutations_dict[n]),len(permutations_dict[n])))

            for i in range(len(permutations_dict[n])):
                for j in range(len(permutations_dict[n])):
                    evaluation_matrix[i,j]=sum(np.equal(labels_dict[n][i], labels_dict[n][j]))

            evaluation_matrices[n] = evaluation_matrix/4
            print("calculated evaluation matrix for n=",n)
            
        return permutations_dict, labels_dict, evaluation_matrices
    
    else:
        return permutations_dict
    
    
def plot_data_mc(model_prediction_prefit, model_prediction_postfit, data, config):
    '''
    Display data-MC comparison plots. Modeled using cabinetry's visualize.data_mc but displays in a less cumbersome way (and shows prefit and postfit plots alongside one another.
    
    Args:
        model_prediction_prefit (model_utils.ModelPrediction): prefit model prediction to show
        model_prediction_postfit (model_utils.ModelPrediction): postfit model prediction to show
        data (List[float]): data to include in visualization
        config (Optional[Dict[str, Any]], optional): cabinetry configuration needed for binning and axis labels
    
    Returns:
        figs: list of matplotlib Figures containing prefit and postfit plots 
    '''
    
    n_bins_total = sum(model_prediction_prefit.model.config.channel_nbins.values())
    if len(data) != n_bins_total:
        data = data[:n_bins_total]

    data_yields = [data[model_prediction_prefit.model.config.channel_slices[ch]] 
                   for ch in model_prediction_prefit.model.config.channels]
    
    channels = model_prediction_prefit.model.config.channels
    channel_indices = [model_prediction_prefit.model.config.channels.index(ch) for ch in channels]
    
    figs = []
    for i_chan, channel_name in zip(channel_indices, channels):
        histogram_dict_list_prefit = []  # one dict per region/channel
        histogram_dict_list_postfit = []  # one dict per region/channel

        regions = [reg for reg in config["Regions"] if reg["Name"] == channel_name]
        region_dict = regions[0]
        bin_edges = np.asarray(region_dict["Binning"])
        variable = region_dict.get("Variable", "bin")
        
        for i_sam, sample_name in enumerate(model_prediction_prefit.model.config.samples):
            histogram_dict_list_prefit.append(
                {"label": sample_name, "isData": False, 
                 "yields": model_prediction_prefit.model_yields[i_chan][i_sam], "variable": variable}
            )
            histogram_dict_list_postfit.append(
                {"label": sample_name, "isData": False, 
                 "yields": model_prediction_postfit.model_yields[i_chan][i_sam], "variable": variable}
            )
            
        # add data sample
        histogram_dict_list_prefit.append(
            {"label": "Data", "isData": True, "yields": data_yields[i_chan], "variable": variable}
        )
        histogram_dict_list_postfit.append(
            {"label": "Data", "isData": True, "yields": data_yields[i_chan], "variable": variable}
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
                mc_histograms_yields_prefit.append(histogram_dict_list_prefit[i]["yields"])
                mc_histograms_yields_postfit.append(histogram_dict_list_postfit[i]["yields"])
                mc_labels.append(histogram_dict_list_prefit[i]["label"])
                
        mpl.style.use("seaborn-v0_8-colorblind")
    
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[1,1])

        # increase font sizes
        for item in (
            [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label, ax3.yaxis.label, ax4.xaxis.label, ax4.yaxis.label]
            + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()
            + ax3.get_yticklabels() + ax4.get_xticklabels() + ax4.get_yticklabels()
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
            mc_container = ax1.bar(bin_centers, mc_histograms_yields_prefit[i], width=bin_width, bottom=total_yield_prefit)
            mc_containers_prefit.append(mc_container)
            mc_container = ax3.bar(bin_centers, mc_histograms_yields_postfit[i], width=bin_width, bottom=total_yield_postfit)
            mc_containers_postfit.append(mc_container)

            # add a black line on top of each sample
            line_x = [y for y in bin_edges for _ in range(2)][1:-1]
            line_y = [y for y in (mc_histograms_yields_prefit[i] + total_yield_prefit) for _ in range(2)]
            ax1.plot(line_x, line_y, "-", color="black", linewidth=0.5)
            line_y = [y for y in (mc_histograms_yields_postfit[i] + total_yield_postfit) for _ in range(2)]
            ax3.plot(line_x, line_y, "-", color="black", linewidth=0.5)

            total_yield_prefit += mc_histograms_yields_prefit[i]
            total_yield_postfit += mc_histograms_yields_postfit[i]
            
        # add total MC uncertainty
        total_model_unc_prefit = np.asarray(model_prediction_prefit.total_stdev_model_bins[i_chan][-1])
        mc_unc_container_prefit = ax1.bar(bin_centers, 2*total_model_unc_prefit, width=bin_width, 
                                          bottom=total_yield_prefit-total_model_unc_prefit, fill=False, linewidth=0, 
                                          edgecolor="gray", hatch=3 * "/",)
        total_model_unc_postfit = np.asarray(model_prediction_postfit.total_stdev_model_bins[i_chan][-1])
        mc_unc_container_postfit = ax3.bar(bin_centers, 2*total_model_unc_postfit, width=bin_width, 
                                           bottom=total_yield_postfit-total_model_unc_postfit, fill=False, linewidth=0, 
                                           edgecolor="gray", hatch=3 * "/",)
    
        # plot data
        data_container_prefit = ax1.errorbar(bin_centers, data_histogram_yields, yerr=data_histogram_stdev, fmt="o", color="k")
        data_container_postfit = ax3.errorbar(bin_centers, data_histogram_yields, yerr=data_histogram_stdev, fmt="o", color="k")
        
        # ratio plots
        ax2.plot([bin_left_edges[0], bin_right_edges[-1]], [1, 1], "--", color="black", linewidth=1)  # reference line along y=1
        ax4.plot([bin_left_edges[0], bin_right_edges[-1]], [1, 1], "--", color="black", linewidth=1)  # reference line along y=1

        n_zero_pred_prefit = sum(total_yield_prefit == 0.0)  # number of bins with zero predicted yields
        if n_zero_pred_prefit > 0:
            log.warning(f"(PREFIT) predicted yield is zero in {n_zero_pred_prefit} bin(s), excluded from ratio plot")
        nonzero_model_yield_prefit = total_yield_prefit != 0.0
        if np.any(total_yield_prefit < 0.0):
            raise ValueError(f"(PREFIT) {label_prefit} total model yield has negative bin(s): {total_yield_prefit.tolist()}")
        
        n_zero_pred_postfit = sum(total_yield_postfit == 0.0)  # number of bins with zero predicted yields
        if n_zero_pred_postfit > 0:
            log.warning(f"(POSTFIT) predicted yield is zero in {n_zero_pred_postfit} bin(s), excluded from ratio plot")
        nonzero_model_yield_postfit = total_yield_postfit != 0.0
        if np.any(total_yield_postfit < 0.0):
            raise ValueError(f"(POSTFIT) {label_postfit} total model yield has negative bin(s): {total_yield_postfit.tolist()}")
        
        # add uncertainty band around y=1
        rel_mc_unc_prefit = total_model_unc_prefit / total_yield_prefit
        rel_mc_unc_postfit = total_model_unc_postfit / total_yield_postfit
        # do not show band in bins where total model yield is 0
        ax2.bar(bin_centers[nonzero_model_yield_prefit], 2*rel_mc_unc_prefit[nonzero_model_yield_prefit], 
                width=bin_width[nonzero_model_yield_prefit], bottom=1.0-rel_mc_unc_prefit[nonzero_model_yield_prefit], 
                fill=False, linewidth=0, edgecolor="gray", hatch=3 * "/")
        ax4.bar(bin_centers[nonzero_model_yield_postfit], 2*rel_mc_unc_postfit[nonzero_model_yield_postfit], 
                width=bin_width[nonzero_model_yield_postfit], bottom=1.0-rel_mc_unc_postfit[nonzero_model_yield_postfit], 
                fill=False, linewidth=0, edgecolor="gray", hatch=3 * "/")
        
        # data in ratio plot
        data_model_ratio_prefit = data_histogram_yields / total_yield_prefit
        data_model_ratio_unc_prefit = data_histogram_stdev / total_yield_prefit
        data_model_ratio_postfit = data_histogram_yields / total_yield_postfit
        data_model_ratio_unc_postfit = data_histogram_stdev / total_yield_postfit
        # mask data in bins where total model yield is 0
        ax2.errorbar(bin_centers[nonzero_model_yield_prefit], data_model_ratio_prefit[nonzero_model_yield_prefit], 
                     yerr=data_model_ratio_unc_prefit[nonzero_model_yield_prefit], fmt="o", color="k")
        ax4.errorbar(bin_centers[nonzero_model_yield_postfit], data_model_ratio_postfit[nonzero_model_yield_postfit], 
                     yerr=data_model_ratio_unc_postfit[nonzero_model_yield_postfit], fmt="o", color="k")
        
        # get the highest single bin yield, from the sum of MC or data
        y_max = max(max(np.amax(total_yield_prefit), np.amax(data_histogram_yields)),
                    max(np.amax(total_yield_postfit), np.amax(data_histogram_yields)))

        ax1.set_ylim([0, y_max * 1.5])  # 50% headroom
        ax3.set_ylim([0, y_max * 1.5])  # 50% headroom
    
        # figure label (region name)
        at = mpl.offsetbox.AnchoredText(label_prefit, loc="upper left", frameon=False, prop={"fontsize": "large", "linespacing": 1.5})
        ax1.add_artist(at)
        at = mpl.offsetbox.AnchoredText(label_postfit, loc="upper left", frameon=False, prop={"fontsize": "large", "linespacing": 1.5})
        ax3.add_artist(at)
        
        # MC contributions in inverse order, such that first legend entry corresponds to
        # the last (highest) contribution to the stack
        all_containers = mc_containers_prefit[::-1] + [mc_unc_container_prefit, data_container_prefit]
        all_labels = mc_labels[::-1] + ["Uncertainty", data_label]
        ax1.legend(all_containers, all_labels, frameon=False, fontsize="large", loc="upper right")
        all_containers = mc_containers_postfit[::-1] + [mc_unc_container_postfit, data_container_postfit]
        ax3.legend(all_containers, all_labels, frameon=False, fontsize="large", loc="upper right")
        
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
    