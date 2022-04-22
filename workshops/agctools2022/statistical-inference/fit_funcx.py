import argparse
import json
from concurrent.futures import as_completed
from pathlib import Path

import pyhf
from funcx import FuncXClient, FuncXExecutor
from pyhf.contrib.utils import download


def prepare_workspace(data, backend):
    import pyhf

    pyhf.set_backend(backend)

    return pyhf.Workspace(data)


def infer_hypotest(workspace, metadata, patches, backend):
    import time

    import pyhf

    pyhf.set_backend(backend)

    tick = time.time()
    model = workspace.model(
        patches=patches,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = workspace.data(model)
    test_poi = 1.0
    CLs_obs, CLs_exp_band = pyhf.infer.hypotest(
        test_poi, data, model, return_expected_set=True, test_stat="qtilde"
    )
    fit_time = time.time() - tick
    return {
        "metadata": metadata,
        "CLs_obs": float(CLs_obs),
        "CLs_exp": [float(cls_exp) for cls_exp in CLs_exp_band],
        "fit_time": fit_time,
    }


def main(args):
    if args.config_file is not None:
        with open(args.config_file, "r") as infile:
            config = json.load(infile)

    backend = args.backend

    pallet_path = Path(config["input_prefix"]).joinpath(config["pallet_name"])

    # locally get pyhf pallet for analysis
    if not pallet_path.exists():
        download(config["pallet_url"], pallet_path)

    analysis_name = config["analysis_name"]
    analysis_prefix_str = "" if analysis_name is None else f"{analysis_name}_"
    if config["analysis_dir"] is not None:
        pallet_path = pallet_path.joinpath(config["analysis_dir"])

    with open(
        pallet_path.joinpath(f"{analysis_prefix_str}BkgOnly.json")
    ) as bkgonly_json:
        bkgonly_workspace = json.load(bkgonly_json)

    # Initialize funcX client
    fxc = FuncXClient()
    fx = FuncXExecutor(fxc, batch_enabled=True)

    with open("endpoint_id.txt") as endpoint_file:
        pyhf_endpoint = str(endpoint_file.read().rstrip())

    # execute background only workspace
    prepare_task_future = fx.submit(
        prepare_workspace, bkgonly_workspace, backend, endpoint_id=pyhf_endpoint
    )

    # Read patchset in while background only workspace running
    with open(
        pallet_path.joinpath(f"{analysis_prefix_str}patchset.json")
    ) as patchset_json:
        patchset = pyhf.PatchSet(json.load(patchset_json))

    workspace = prepare_task_future.result()
    message = "# Background Workspace Constructed"
    print("-" * len(message))
    print(message)
    print("-" * len(message))

    # execute patch fits across workers and retrieve them when done
    n_patches = len(patchset.patches)
    futures = []
    results = {}
    for patch_idx in range(n_patches):
        patch = patchset.patches[patch_idx]
        futures.append(
            fx.submit(
                infer_hypotest,
                workspace,
                patch.metadata,
                [patch.patch],
                backend,
                endpoint_id=pyhf_endpoint,
            )
        )

    for task in as_completed(futures):
        task_result = task.result()
        results[task_result["metadata"]["name"]] = {
            "mass_hypotheses": task_result["metadata"]["values"],
            "CLs_obs": task_result["CLs_obs"],
            "CLs_exp": task_result["CLs_exp"],
            "fit_time": task_result["fit_time"],
        }
        print(
            f'{task_result["metadata"]["name"]}: {results[task_result["metadata"]["name"]]}'
        )

    print("-" * len(message))

    with open("results.json", "w") as results_file:
        results_file.write(json.dumps(results, sort_keys=True, indent=2))


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="configuration arguments provided at run time from the CLI"
    )
    cli_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        type=str,
        default=None,
        help="config file",
    )
    cli_parser.add_argument(
        "-b",
        "--backend",
        dest="backend",
        type=str,
        default="numpy",
        help="pyhf backend str alias",
    )
    args, unknown = cli_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False)
    args = parser.parse_args()

    main(args)
