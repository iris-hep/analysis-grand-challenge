import json
import numpy as np
import os
from pathlib import Path
import tqdm
import urllib


try:
    from servicex import ServiceXDataset
except ImportError:
    # if servicex is not available, ServiceXDatasetGroup cannot be used
    # this is fine for worker nodes: only needed where main notebook is executed
    pass


# If local_data_cache is a writable path, this function will download any missing file into it and
# then return file paths corresponding to these local copies.
def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name="", local_data_cache=None, input_from_eos=False, xcache_atlas_prefix=None):
    if af_name == "ssl-dev":
        if use_xcache:
            raise RuntimeError("`use_xcache` and `af_name='ssl-dev'` are incompatible. Please only use one of them.")
        if local_data_cache is not None:
            raise RuntimeError("`af_name='ssl-dev'` and `local_data_cache` are incompatible. Please only use one of them.")
        if input_from_eos:
            raise RuntimeError("`af_name='ssl-dev'` and `input_from_eos` are incompatible. Please only use one of them.")

    if input_from_eos:
        if local_data_cache:
            # download relies on https, EOS files use xrootd
            raise RuntimeError("`input_from_eos` and `local_data_cache` are incompatible. Please only use one of them.")
        if use_xcache:
            raise RuntimeError("`input_from_eos` and `use_xcache` are incompatible. Please only use one of them.")

    if local_data_cache is not None:
        local_data_cache = Path(local_data_cache)
        if not local_data_cache.exists() or not os.access(local_data_cache, os.W_OK):
            raise RuntimeError(f"local_data_cache directory {local_data_cache} does not exist or is not writable.")

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

    # nicer labels for plots
    process_labels = {
        "ttbar": r"$t\bar{t}$",
        "single_top_s_chan": r"$s$-channel single top",
        "single_top_t_chan": r"$t$-channel single top",
        "single_top_tW": r"$tW$",
        "wjets": r"$W$+jets"
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
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://xcache") for f in file_paths]
            elif af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/AGC", "/data/alheld/AGC/datasets") for f in file_paths]
            elif input_from_eos:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/AGC/nanoAOD",
                                        "root://eospublic.cern.ch//eos/opendata/cms/upload/agc/1.0.0/") for f in file_paths]

            if xcache_atlas_prefix is not None:
                # prepend xcache to paths
                file_paths = [xcache_atlas_prefix + f for f in file_paths]

            if local_data_cache is not None:
                local_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/", f"{local_data_cache.absolute()}/") for f in file_paths]
                for remote, local in zip(file_paths, local_paths):
                    if not Path(local).exists():
                        download_file(remote, local)
                file_paths = local_paths
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process], "process_label": process_labels[process]}
            # coffea now wants file list entries as dict instead of list with format {path_1: treename, path_2: treename}
            file_paths = dict(zip(file_paths, ["Events"]*len(file_paths)))
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset
