#!/usr/bin/env python

# Taken from https://github.com/lukasheinrich/sbottom-lhood-pubnote-code/blob/master/scripts/harvest.py

import json


def make_harvest_from_result(result):
    # These keys are not all needed, but are the extras that are coming along
    # for the ride until they can be downsized.
    return {
        "CLs": result["CLs_obs"],
        "CLsexp": result["CLs_exp"][2],
        "clsd1s": result["CLs_exp"][1],
        "clsd2s": result["CLs_exp"][0],
        "clsu1s": result["CLs_exp"][3],
        "clsu2s": result["CLs_exp"][4],
        "covqual": 3,
        "dodgycov": 0,
        "excludedXsec": -999007,
        "expectedUpperLimit": -1,
        "expectedUpperLimitMinus1Sig": -1,
        "expectedUpperLimitMinus2Sig": -1,
        "expectedUpperLimitPlus1Sig": -1,
        "expectedUpperLimitPlus2Sig": -1,
        "fID": -1,
        "failedcov": 0,
        "failedfit": 0,
        "failedp0": 0,
        "failedstatus": 0,
        "fitstatus": 0,
        "mn1": result["mass_hypotheses"][0],
        "mn2": result["mass_hypotheses"][1],
        "mode": -1,
        "msb": 60,  # Is this right? Check with Giordon
        "nexp": -1,
        "nofit": 0,
        "p0": 0,
        "p0d1s": -1,
        "p0d2s": -1,
        "p0exp": -1,
        "p0u1s": -1,
        "p0u2s": -1,
        "p1": 0,
        "seed": 0,
        "sigma0": -1,
        "sigma1": -1,
        "upperLimit": -1,
        "upperLimitEstimatedError": -1,
        "xsec": -999007,
    }


def main(target_file="results.json"):
    with open(target_file) as read_file:
        results = json.load(read_file)

    harvests = {
        key: make_harvest_from_result(values) for key, values in results.items()
    }

    with open("harvests.json", "w") as write_file:
        json.dump(harvests, write_file, sort_keys=True, indent=2)


# @click.command()
# @click.argument("master_input_dir", default="data/inputs")
# @click.argument("master_output_dir", default="data/outputs")
# def main(master_input_dir, master_output_dir):
#     pattern = re.compile("sbottom_(\d+)_(\d+)_(\d+)")
#     os.makedirs(os.path.join(master_output_dir, "harvest"), exist_ok=True)
#     for region in ["A", "B", "C"]:
#         harvest = []

#         thedir = os.path.join(
#             master_output_dir,
#             f"results/Region{region}",
#         )

#         files = f"{thedir}/result_sbottom_*_*_*.json"
#         print(files)
#         for fname in glob.glob(files):
#             # print(fname)
#             result = json.load(open(fname))
#             m = pattern.search(fname)
#             masses = list(map(int, m.groups()))
#             # only use 60 GeV
#             if masses[2] != 60:
#                 continue
#             harvest.append(make_harvest_from_result(result, masses))
#         json.dump(
#             harvest,
#             open(
#                 os.path.join(
#                     master_output_dir,
#                     "harvest",
#                     f"Region{region}.json",
#                 ),
#                 "w+",
#             ),
#             sort_keys=True,
#             indent=2,
#         )


if __name__ == "__main__":
    main()
