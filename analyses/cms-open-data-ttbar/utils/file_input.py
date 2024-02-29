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
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


def tqdm_urlretrieve_hook(t):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py ."""
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to


def download_file(url, out_file):
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=out_path.name) as t:
        urllib.request.urlretrieve(url, out_path.absolute(), reporthook=tqdm_urlretrieve_hook(t))


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