import json
import os
import time

from .config import config

def track_metrics(metrics, fileset, exec_time, USE_DASK, USE_SERVICEX, N_FILES_MAX_PER_SAMPLE, USE_INFERENCE, USE_TRITON):

    file_name = fileset["ttbar__nominal"]["files"][0]
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

    metrics.update({
        "walltime": exec_time,
        "num_workers": config["benchmarking"]["NUM_CORES"],
        "af": config["benchmarking"]["AF_NAME"],
        "dataset_source": dataset_source,
        "use_dask": USE_DASK,
        "use_servicex": USE_SERVICEX,
        "systematics": config["benchmarking"]["SYSTEMATICS"],
        "n_files_max_per_sample": N_FILES_MAX_PER_SAMPLE,
        "cores_per_worker": config["benchmarking"]["CORES_PER_WORKER"],
        "chunksize": config["benchmarking"]["CHUNKSIZE"],
        "disable_processing": config["benchmarking"]["DISABLE_PROCESSING"],
        "io_file_percent": config["benchmarking"]["IO_FILE_PERCENT"],
        "agc_version": "main",
        "use_inference": USE_INFERENCE,
        "use_triton": USE_TRITON
    })

    # save metrics to disk
    if not os.path.exists("metrics"):
        os.makedirs("metrics")
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    af_name = metrics["af"]
    metric_file_name = f"metrics/{af_name}-{timestamp}.json"
    with open(metric_file_name, "w") as f:
        f.write(json.dumps(metrics))

    print(f"metrics saved as {metric_file_name}")
    #print(f"event rate per worker (full execution time divided by NUM_CORES={NUM_CORES}): {metrics['entries'] / NUM_CORES / exec_time / 1_000:.2f} kHz")
    print(f"event rate per worker (pure processtime): {metrics['entries'] / metrics['processtime'] / 1_000:.2f} kHz")
    print(f"amount of data read: {metrics['bytesread']/1000**2:.2f} MB (note that this can be buggy: https://github.com/CoffeaTeam/coffea/issues/717)")
