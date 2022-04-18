from collections import defaultdict
import json


def get_paths(process, recids):
    if not isinstance(recids, list):
        recids = [recids]

    all_files = []
    for recid in recids:
        with open(f"{process}/{str(recid)}.txt") as f:
            files = f.readlines()
        if process == "data":
            prefix_eos = "root://eospublic.cern.ch//eos/opendata/cms/"
        else:
            prefix_eos = "root://eospublic.cern.ch//eos/opendata/cms/mc/"
        prefix_unl = "root://xrootd-local.unl.edu:1094//store/user/AGC/datasets/"
        all_files += [f.strip().replace(prefix_eos, prefix_unl) for f in files]

    return all_files


def update_dict(file_dict, process, sample, recid):
    files = get_paths(process, recid)
    file_dict[process].update({sample: files})
    return file_dict


def write_to_file(file_dict, path):
    with open(path, "w") as f:
        f.write(json.dumps(file_dict, sort_keys=False, indent=4))
        f.write("\n")


if __name__ == "__main__":
    file_dict = defaultdict(dict)

    # ttbar
    update_dict(file_dict, "ttbar", "nominal", 19980)  # can also add 19981
    update_dict(file_dict, "ttbar", "scaledown", 19983)
    update_dict(file_dict, "ttbar", "scaleup", 19985)
    update_dict(file_dict, "ttbar", "ME_var", 19978)
    update_dict(file_dict, "ttbar", "PS_var", 19999)

    # single top
    update_dict(file_dict, "single_top", "s_chan", 19394)
    update_dict(file_dict, "single_top", "t_chan", [19406, 19408])
    update_dict(file_dict, "single_top", "tW", [19412, 19419])

    # W+jets
    update_dict(file_dict, "wjets", "nominal", 20547)  # can also add 20548

    # data
    update_dict(file_dict, "data", "nominal", [24119, 24120])

    write_to_file(file_dict, "ntuples.json")
