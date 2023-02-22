import asyncio
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot

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
    plt.rcParams["font.size"] = 12
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
    with open("ntuples_merged.json") as f:
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