import asyncio
import json

import cabinetry
from cabinetry.contrib import histogram_reader
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot
from servicex import ServiceXDataset
import numpy as np

def get_client(af="coffea_casa"):
    if af == "coffea_casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from htcdaskgateway import HTCGateway

        gateway = HTCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: https://dask-gateway.fnal.gov/clusters/{str(cluster.name)}/status")

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


def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name="", input_from_eos=False, xcache_atlas_prefix=None):
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
            elif af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/AGC", "/data/alheld/AGC/datasets") for f in file_paths]
            elif input_from_eos:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/AGC/nanoAOD", "root://eospublic.cern.ch//eos/opendata/cms/upload/agc/1.0.0/") for f in file_paths]
            if xcache_atlas_prefix is not None:
                # prepend xcache to paths
                file_paths = [xcache_atlas_prefix + f for f in file_paths]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


def save_histograms(all_histograms, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    pseudo_data = (all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 2  +\
        all_histograms[:, :, "wjets", "nominal"] + all_histograms[:, :, "single_top_t_chan", "nominal"]

    with uproot.recreate(filename) as f:
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[:, region]
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[:, region, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[:, region, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[:, region, sample_name, variation_name]

            # ttbar modeling
            f[f"{region}_ttbar_ME_var"] = all_histograms[:, region, "ttbar", "ME_var"]
            f[f"{region}_ttbar_PS_var"] = all_histograms[:, region, "ttbar", "PS_var"]

            f[f"{region}_ttbar_scaledown"] = all_histograms[:, region, "ttbar", "scaledown"]
            f[f"{region}_ttbar_scaleup"] = all_histograms[:, region, "ttbar", "scaleup"]

            # W+jets scale
            f[f"{region}_wjets_scale_var_down"] = all_histograms[:, region, "wjets", "scale_var_down"]
            f[f"{region}_wjets_scale_var_up"] = all_histograms[:, region, "wjets", "scale_var_up"]


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

        try:
            # default matching for when ServiceX doesn't abbreviate names
            parent_file_urls = np.array([f.file for f in all_files])

            # order is not retained after transform, so we can match files to their parent files using the filename
            # (replacing / with : to mitigate servicex filename convention )
            parent_key = np.array([np.where(parent_file_urls==self.filelist[i][0].replace("/",":"))[0][0]
                                   for i in range(len(self.filelist))])
        except IndexError:
            # fallback solution that relies splitting via the port (name only changes before that)
            # probably not very stable and general! this may fail - please report back if you observe that happening
            # TODO: find something more stable
            parent_file_urls = np.asarray([f.replace(":", "/").split("1094//")[-1] for f in np.array([f.file for f in all_files])])
            parent_key = np.array([np.where(parent_file_urls==self.filelist[i][0].split("1094//")[-1])[0][0]
                                   for i in range(len(self.filelist))])

        files_per_process = {}
        for i, process in enumerate(self.fileset):
            # update files for each process
            files_per_process.update({process: all_files[parent_key[self.filelist[:,1]==process]]})

        return files_per_process


def get_cabinetry_rebinning_router(config, rebinning):
    # perform re-binning in cabinetry by providing a custom function reading histograms
    # will eventually be replaced via https://github.com/scikit-hep/cabinetry/issues/412
    rebinning_router = cabinetry.route.Router()

    # this reimplements some of cabinetry.templates.collect
    general_path = config["General"]["InputPath"]
    variation_path = config["General"].get("VariationPath", None)

    # define a custom template builder function that is executed for data samples
    @rebinning_router.register_template_builder()
    def build_data_hist(region, sample, systematic, template):
        # get path to histogram
        histo_path = cabinetry.templates.collector._histo_path(general_path, variation_path, region, sample, systematic, template)
        h = hist.Hist(histogram_reader.with_uproot(histo_path))  # turn from boost-histogram into hist
        # perform re-binning
        h = h[rebinning]
        return h

    return rebinning_router


def dataset_source(file_name):
    if file_name.startswith("/data"):
        dataset_source = "/data"
    elif "xcache.af.uchicago.edu" in file_name:
        dataset_source = "xcache.af.uchicago.edu"
    elif "red-xcache1.unl.edu" in file_name:
        dataset_source = "red-xcache1.unl.edu"
    elif "eospublic" in file_name:
        dataset_source = "EOS"
    elif "xrootd-local.unl.edu" in file_name:
        dataset_source = "UNL"
    else:
        dataset_source = "unknown"
    return dataset_source
