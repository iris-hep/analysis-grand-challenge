from collections import namedtuple

import numpy as np
from descartes import PolygonPatch
from shapely.geometry.polygon import Polygon

from exclusion.interpolate import main as interpolate_main


def harvest_from_result(results_dict):
    harvests = {
        key: {
            "CLs": values["CLs_obs"],
            "CLsexp": values["CLs_exp"][2],
            "clsd1s": values["CLs_exp"][1],
            "clsd2s": values["CLs_exp"][0],
            "clsu1s": values["CLs_exp"][3],
            "clsu2s": values["CLs_exp"][4],
            "mn1": values["mass_hypotheses"][0],
            "mn2": values["mass_hypotheses"][1],
            "failedstatus": 0,
            "upperLimit": -1,
            "expectedUpperLimit": -1,
        }
        for key, values in results_dict.items()
    }

    return harvests


def make_interpolated_results(results):
    # d = {
    #     "figureOfMerit": "CLsexp",
    #     "modelDef": "mn2,mn1",
    #     "ignoreTheory": True,
    #     "ignoreUL": True,
    #     "debug": False,
    # }
    # args = namedtuple("Args", d.keys())(**d)
    # multiplex_data = multiplex_main(args, inputDataList=dataList).to_dict(
    #     orient="records"
    # )

    kwargs = {
        "nominalLabel": "Nominal",
        "xMin": None,
        "xMax": None,
        "yMin": None,
        "yMax": None,
        "smoothing": "0.02",
        # "smoothing": "0.07",
        "areaThreshold": 0,
        "xResolution": 100,
        "yResolution": 100,
        "xVariable": "mn1",
        "yVariable": "mn2",
        "closedBands": False,
        "forbiddenFunction": "x",
        "debug": False,
        "logX": False,
        "logY": False,
        "noSig": False,
        "interpolation": "multiquadric",
        # "interpolationEpsilon": 0,
        "interpolationEpsilon": 0.05,
        "level": 1.64485362695,
        "useROOT": False,
        "sigmax": 5,
        "useUpperLimit": False,
        "ignoreUncertainty": False,
        "fixedParamsFile": "",
    }
    args = namedtuple("Args", kwargs.keys())(**kwargs)
    # r = interpolate_main(args, multiplex_data)

    harvests = harvest_from_result(results)

    return interpolate_main(args, inputData=harvests)


def plot_contour(ax, results, **kwargs):

    if kwargs.get("show_points", False):
        mass_ranges = np.asarray(
            [values["mass_hypotheses"] for _, values in results.items()]
        ).T
        ax.scatter(*mass_ranges, s=20, alpha=0.2)

    if kwargs.get("show_interpolated", False):
        interpolated_bands = make_interpolated_results(results)
        if interpolated_bands is None:
            print("ERROR: interpolation failed")
            return 1

        if "Band_1s_0" not in interpolated_bands:
            print("ERROR: Band_1s_0 not included in interpolation bands")
            return 1

        # Expand the transverse of the array into the x,y components of the sequence
        # for the shell arg of Polygon
        ax.add_patch(
            PolygonPatch(
                Polygon(np.stack([*interpolated_bands["Band_1s_0"].T]).T),
                alpha=0.5,
                facecolor=kwargs.get("color", "steelblue"),
                label=r"Expected Limit ($\pm1\sigma$)",
            ),
        )

        # Expand the transverse of the array into x,y args of plot
        ax.plot(
            *interpolated_bands["Exp_0"].T,
            color="black",
            linestyle="dashed",
            alpha=0.5,
        )

        ax.plot(
            *interpolated_bands["Obs_0"].T,
            color="maroon",
            linewidth=2,
            linestyle="solid",
            alpha=0.5,
            label="Observed Limit",
        )

    return ax
