# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ATLAS Open Data $H\rightarrow ZZ^\star$ with `ServiceX`, `coffea`, `cabinetry` & `pyhf`

# %%
import re
import time

import awkward as ak
import cabinetry
from func_adl import ObjectStream
from func_adl_servicex import ServiceXSourceUpROOT
import hist
import mplhep
import numpy as np
import pyhf
import uproot
from servicex import ServiceXDataset

from coffea import processor
from coffea.nanoevents.schemas.base import BaseSchema
import utils
from utils import infofile  # contains cross-section information

import vector
vector.register_awkward()

utils.clean_up()  # delete output from previous runs of notebook (optional)
utils.set_logging()  # configure logging output

# %%
# set some global settings

# chunk size to use
CHUNKSIZE = 500_000

# scaling for local setups with FuturesExecutor
NUM_CORES = 4

# ServiceX behavior: ignore cache with repeated queries
IGNORE_CACHE = False

# %% [markdown]
# ## Introduction
#
# We are going to use the ATLAS Open Data for this demonstration, in particular a $H\rightarrow ZZ^\star$ analysis. Find more information on the [ATLAS Open Data documentation](http://opendata.atlas.cern/release/2020/documentation/physics/FL2.html) and in [ATL-OREACH-PUB-2020-001](https://cds.cern.ch/record/2707171). The datasets used are [10.7483/OPENDATA.ATLAS.2Y1T.TLGL](http://doi.org/10.7483/OPENDATA.ATLAS.2Y1T.TLGL). The material in this notebook is based on the [ATLAS Open Data notebooks](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata), a [PyHEP 2021 ServiceX demo](https://github.com/gordonwatts/pyhep-2021-SX-OpenDataDemo), and [Storm Lin's adoption](https://github.com/stormsomething/CoffeaHZZAnalysis) of this analysis.
#
# This notebook is meant as a **technical demonstration**. In particular, the systematic uncertainties defined are purely to demonstrate technical aspects of realistic workflows, and are not meant to be meaningful physically. The fit performed to data consequently also only demonstrate technical aspects. If you are interested about the physics of $H\rightarrow ZZ^\star$, check out for example the actual ATLAS cross-section measurement: [Eur. Phys. J. C 80 (2020) 942](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGG-2018-29/).
#
# This notebook implements most of the analysis pipeline shown in the following picture, using the tools also mentioned there:
# ![ecosystem visualization](utils/ecosystem.png)

# %% [markdown]
# ### Tools and packages used in this example
#
# The notebook showcases:
# - data delivery with `ServiceX`
# - event / column selection with `func_adl`
# - data handling with `awkward-array`
# - histogram production with `coffea`
# - histogram handling with `hist`
# - visualization with `mplhep`, `hist` & `matplotlib`
# - ROOT file handling with `uproot`
# - statistical model construction with `cabinetry`
# - statistical inference with `pyhf`

# %% [markdown]
# ### High-level strategy
#
# We will define which files to process, set up a query with `func_adl` to extract data provided by `ServiceX`, and use `coffea` to construct histograms.
# Those histograms will be saved with `uproot`, and then assembled into a statistical model with `cabinetry`.
# Following that, we perform statistical inference with `pyhf`.

# %% [markdown]
# ### Required setup for this notebook
#
# If you are running on the coffea-casa Open Data instance, ServiceX credentials are automatically available to you.
# Otherwise you will need to set those up.
# Create a file `servicex.yaml` in your home directory, or the place this notebook is located in.
#
# See this [talk by KyungEon](https://indico.cern.ch/event/1076231/contributions/4560404/) and the [ServiceX doc](https://servicex.readthedocs.io/en/latest/user/getting-started/) for more information.

# %% [markdown]
# ## Files to process
#
# To get started, we define which files are going to be processed in this notebook.
# We also set some information for histogramming that will be used subsequently.

# %%
prefix = (
    "http://xrootd-local.unl.edu:1094//store/user/AGC/ATLAS_HZZ/"
)

# labels for combinations of datasets
z_ttbar = r"Background $Z,t\bar{t}$"
zzstar = r"Background $ZZ^{\star}$"
signal = r"Signal ($m_H$ = 125 GeV)"

input_files = {
    "Data": [
        prefix + "data_A.4lep.root",
        prefix + "data_B.4lep.root",
        prefix + "data_C.4lep.root",
        prefix + "data_D.4lep.root",
    ],
    z_ttbar: [
        prefix + "mc_361106.Zee.4lep.root",
        prefix + "mc_361107.Zmumu.4lep.root",
        prefix + "mc_410000.ttbar_lep.4lep.root",
    ],
    zzstar: [prefix + "mc_363490.llll.4lep.root"],
    signal: [
        prefix + "mc_345060.ggH125_ZZ4lep.4lep.root",
        prefix + "mc_344235.VBFH125_ZZ4lep.4lep.root",
        prefix + "mc_341964.WH125_ZZ4lep.4lep.root",
        prefix + "mc_341947.ZH125_ZZ4lep.4lep.root",
    ],
}

# information for histograms
bin_edge_low = 80  # 80 GeV
bin_edge_high = 250  # 250 GeV
num_bins = 34


# %% [markdown]
# ## Setting up a query with `func_adl`
#
# We are using `func_adl` for event & column selection, and make a datasource with the query built by `get_lepton_query`.
#
# A list of all available columns in the input files can be found in the [ATLAS documentation of branches](http://opendata.atlas.cern/release/2020/documentation/datasets/dataset13.html).
#
# <span style="color:darkgreen">**Systematic uncertainty added:**</span> scale factor variation, applied already at event selection stage. Imagine that this could be a calculation that requires a lot of different variables which are no longer needed downstream afterwards, so it makes sense to do it here.

# %%
def get_lepton_query(source: ObjectStream) -> ObjectStream:
    """Performs event selection: require events with exactly four leptons.
    Also select all columns needed further downstream for processing &
    histogram filling.
    """
    return source.Where(lambda event: event.lep_n == 4).Select(
        lambda e: {
            "lep_pt": e.lep_pt,
            "lep_eta": e.lep_eta,
            "lep_phi": e.lep_phi,
            "lep_energy": e.lep_E,
            "lep_charge": e.lep_charge,
            "lep_typeid": e.lep_type,
            "mcWeight": e.mcWeight,
            "scaleFactor": e.scaleFactor_ELE
            * e.scaleFactor_MUON
            * e.scaleFactor_LepTRIGGER
            * e.scaleFactor_PILEUP,
            # scale factor systematic variation example
            "scaleFactorUP": e.scaleFactor_ELE
            * e.scaleFactor_MUON
            * e.scaleFactor_LepTRIGGER
            * e.scaleFactor_PILEUP
            * 1.1,
            "scaleFactorDOWN": e.scaleFactor_ELE
            * e.scaleFactor_MUON
            * e.scaleFactor_LepTRIGGER
            * e.scaleFactor_PILEUP
            * 0.9,
        }
    )


# %% [markdown]
# # Caching the queried datasets with `ServiceX`
#
# Using the queries created with `func_adl`, we are using `ServiceX` to read the ATLAS Open Data files to build cached files with only the specific event information as dictated by the query.

# %%
# dummy dataset on which to generate the query
dummy_ds = ServiceXSourceUpROOT("cernopendata://dummy", "mini", backend_name="uproot")

# tell low-level infrastructure not to contact ServiceX yet, only to
# return the qastle string it would have sent
dummy_ds.return_qastle = True

# create the query
lepton_query = get_lepton_query(dummy_ds)
query = lepton_query.value()

# now we query the files and create a fileset dictionary containing the
# URLs pointing to the queried files

t0 = time.time()

fileset = {}

for ds_name in input_files.keys():
    ds = ServiceXDataset(input_files[ds_name], backend_name="uproot", ignore_cache=IGNORE_CACHE)
    files = ds.get_data_rootfiles_uri(query, as_signed_url=True, title=ds_name)

    fileset[ds_name] = {"files": [f.url for f in files],
                        "metadata": {"dataset_name": ds_name}
                       }

print(f"execution took {time.time() - t0:.2f} seconds")


# %% [markdown]
# We now have a fileset dictionary containing the addresses of the queried files, ready to pass to `coffea`:

# %%
fileset


# %% [markdown]
# ## Processing `ServiceX`-provided data with `coffea`
#
# Event weighting: look up cross-section from a provided utility file, and correctly normalize all events.

# %%
def get_xsec_weight(sample: str) -> float:
    """Returns normalization weight for a given sample."""
    lumi = 10_000  # pb^-1
    xsec_map = infofile.infos[sample]  # dictionary with event weighting information
    xsec_weight = (lumi * xsec_map["xsec"]) / (xsec_map["sumw"] * xsec_map["red_eff"])
    return xsec_weight


# %% [markdown]
# Cuts to apply:
# - two opposite flavor lepton pairs (total lepton charge is 0)
# - lepton types: 4 electrons, 4 muons, or 2 electrons + 2 muons

# %%
def lepton_filter(lep_charge, lep_type):
    """Filters leptons: sum of charges is required to be 0, and sum of lepton types 44/48/52.
    Electrons have type 11, muons have 13, so this means 4e/4mu/2e2mu.
    """
    sum_lep_charge = ak.sum(lep_charge, axis=1)
    sum_lep_type = ak.sum(lep_type, axis=1)
    good_lep_type = ak.any(
        [sum_lep_type == 44, sum_lep_type == 48, sum_lep_type == 52], axis=0
    )
    return ak.all([sum_lep_charge == 0, good_lep_type], axis=0)


# %% [markdown]
# Set up the `coffea` processor. It will apply cuts, calculate the four-lepton invariant mass, and fill a histogram.
#
# <span style="color:darkgreen">**Systematic uncertainty added:**</span> m4l variation, applied in the processor to remaining events. This might instead for example be the result of applying a tool performing a computationally expensive calculation, which should only be run for events where it is needed.

# %%
class HZZAnalysis(processor.ProcessorABC):
    """The coffea processor used in this analysis."""

    def __init__(self):
        pass

    def process(self, events):
        vector.register_awkward()
        # type of dataset being processed, provided via metadata (comes originally from fileset)
        dataset_category = events.metadata["dataset_name"]

        # apply a cut to events, based on lepton charge and lepton type
        events = events[lepton_filter(events.lep_charge, events.lep_typeid)]

        # construct lepton four-vectors
        leptons = ak.zip(
            {"pt": events.lep_pt,
             "eta": events.lep_eta,
             "phi": events.lep_phi,
             "energy": events.lep_energy},
            with_name="Momentum4D",
        )

        # calculate the 4-lepton invariant mass for each remaining event
        # this could also be an expensive calculation using external tools
        mllll = (
            leptons[:, 0] + leptons[:, 1] + leptons[:, 2] + leptons[:, 3]
        ).mass / 1000

        # creat histogram holding outputs, for data just binned in m4l
        mllllhist_data = hist.Hist.new.Reg(
            num_bins,
            bin_edge_low,
            bin_edge_high,
            name="mllll",
            label="$\mathrm{m_{4l}}$ [GeV]",
        ).Weight()  # using weighted storage here for plotting later, but not needed

        # three histogram axes for MC: m4l, category, and variation (nominal and
        # systematic variations)
        mllllhist_MC = (
            hist.Hist.new.Reg(
                num_bins,
                bin_edge_low,
                bin_edge_high,
                name="mllll",
                label="$\mathrm{m_{4l}}$ [GeV]",
            )
            .StrCat([k for k in fileset.keys() if k != "Data"], name="dataset")
            .StrCat(
                ["nominal", "scaleFactorUP", "scaleFactorDOWN", "m4lUP", "m4lDOWN"],
                name="variation",
            )
            .Weight()
        )

        if dataset_category == "Data":
            # create and fill a histogram for m4l
            mllllhist_data.fill(mllll=mllll)

        else:
            # extract the sample name from the filename to calculate x-sec weight
            sample = re.findall(r"mc_\d+\.(.+)\.4lep", events.metadata["filename"])[0]
            basic_weight = get_xsec_weight(sample) * events.mcWeight
            totalWeights = basic_weight * events.scaleFactor

            # calculate systematic variations for weight
            totalWeightsUp = basic_weight * events.scaleFactorUP
            totalWeightsDown = basic_weight * events.scaleFactorDOWN

            # create and fill weighted histograms for m4l: nominal and variations
            mllllhist_MC.fill(
                mllll=mllll,
                dataset=dataset_category,
                variation="nominal",
                weight=totalWeights,
            )

            # scale factor variations
            mllllhist_MC.fill(
                mllll=mllll,
                dataset=dataset_category,
                variation="scaleFactorUP",
                weight=totalWeightsUp,
            )
            mllllhist_MC.fill(
                mllll=mllll,
                dataset=dataset_category,
                variation="scaleFactorDOWN",
                weight=totalWeightsDown,
            )

            # variation in 4-lepton invariant mass
            mllllhist_MC.fill(
                mllll=mllll * 1.01,
                dataset=dataset_category,
                variation="m4lUP",
                weight=totalWeights,
            )
            mllllhist_MC.fill(
                mllll=mllll * 0.99,
                dataset=dataset_category,
                variation="m4lDOWN",
                weight=totalWeights,
            )

        return {"data": mllllhist_data, "MC": mllllhist_MC}

    def postprocess(self, accumulator):
        pass


# %% [markdown]
# ## Producing the desired histograms
#
# Run the processor on data previously gathered by ServiceX, then gather output histograms.

# %%
t0 = time.time()

executor = processor.FuturesExecutor(workers=NUM_CORES)
run = processor.Runner(executor=executor, savemetrics=True, metadata_cache={},
                       chunksize=CHUNKSIZE, schema=BaseSchema)
all_histograms, metrics = run(fileset, "servicex", processor_instance=HZZAnalysis())

print(f"execution took {time.time() - t0:.2f} seconds")

# %% [markdown]
# ## Plotting histograms with `mplhep`, `hist` & `matplotlib`
#
# We can plot some of the histograms we just produced with `mplhep`, `hist` & `matplotlib`.

# %%
# plot histograms with mplhep & hist
mplhep.histplot(
    all_histograms["data"], histtype="errorbar", color="black", label="Data"
)
hist.Hist.plot1d(
    all_histograms["MC"][:, :, "nominal"],
    stack=True,
    histtype="fill",
    color=["purple", "red", "lightblue"],
)

# plot band for MC statistical uncertainties via utility function
# (this uses matplotlib directly)
utils.plot_errorband(bin_edge_low, bin_edge_high, num_bins, all_histograms)

# we are using a small utility function to also save the figure in .png and .pdf
# format, you can find the produced figure in the figures/ folder
utils.save_figure("m4l_stat_uncertainty")

# %% [markdown]
# ## Saving histograms with `uproot`
#
# In order to build a statistical model, we will use `cabinetry`'s support for reading histograms to build a so-called workspace specifying the model.
# We will save the histograms we just created to disk with `uproot`.

# %%
file_name = "histograms.root"
with uproot.recreate(file_name) as f:
    f["data"] = all_histograms["data"]

    f["Z_tt"] = all_histograms["MC"][:, z_ttbar, "nominal"]
    f["Z_tt_SF_up"] = all_histograms["MC"][:, z_ttbar, "scaleFactorUP"]
    f["Z_tt_SF_down"] = all_histograms["MC"][:, z_ttbar, "scaleFactorDOWN"]
    f["Z_tt_m4l_up"] = all_histograms["MC"][:, z_ttbar, "m4lUP"]
    f["Z_tt_m4l_down"] = all_histograms["MC"][:, z_ttbar, "m4lDOWN"]

    f["ZZ"] = all_histograms["MC"][:, zzstar, "nominal"]
    f["ZZ_SF_up"] = all_histograms["MC"][:, zzstar, "scaleFactorUP"]
    f["ZZ_SF_down"] = all_histograms["MC"][:, zzstar, "scaleFactorDOWN"]
    f["ZZ_m4l_up"] = all_histograms["MC"][:, zzstar, "m4lUP"]
    f["ZZ_m4l_down"] = all_histograms["MC"][:, zzstar, "m4lDOWN"]

    f["signal"] = all_histograms["MC"][:, signal, "nominal"]
    f["signal_SF_up"] = all_histograms["MC"][:, signal, "scaleFactorUP"]
    f["signal_SF_down"] = all_histograms["MC"][:, signal, "scaleFactorDOWN"]
    f["signal_m4l_up"] = all_histograms["MC"][:, signal, "m4lUP"]
    f["signal_m4l_down"] = all_histograms["MC"][:, signal, "m4lDOWN"]

# %% [markdown]
# ## Building a workspace and running a fit with `cabinetry` & `pyhf`
#
# Take a look at the `cabinetry` configuration file in `config.yml`.
# It specifies the model to be built.
# In particular, it lists the samples we are going to be using:

# %%
config = cabinetry.configuration.load("config.yml")
config["Samples"]

# %% [markdown]
# It also shows which systematic uncertainties we will apply.
# This includes a new systematic uncertainty defined at this stage.
#
# <span style="color:darkgreen">**Systematic uncertainty added:**</span> $ZZ^\star$ normalization; this does not require any histograms, so we can define it here.

# %%
config["Systematics"]

# %% [markdown]
# The information in the configuration is used to construct a statistical model, the workspace.

# %%
cabinetry.templates.collect(config)
cabinetry.templates.postprocess(config)  # optional post-processing (e.g. smoothing)
ws = cabinetry.workspace.build(config)

# %% [markdown]
# Create a `pyhf` model and extract the data from the workspace. Perform a MLE fit, the results will be reported in the output.

# %%
model, data = cabinetry.model_utils.model_and_data(ws)
fit_results = cabinetry.fit.fit(model, data)

# %% [markdown]
# We can visualize the pulls and correlations.
# `cabinetry` saves this figure by default as a `.pdf`, but here we will use our small utility again to save in both `.png` and `.pdf` format.

# %%
cabinetry.visualize.pulls(
    fit_results, exclude="Signal_norm", close_figure=False, save_figure=False
)
utils.save_figure("pulls")

# %%
cabinetry.visualize.correlation_matrix(
    fit_results, pruning_threshold=0.15, close_figure=False, save_figure=False
)
utils.save_figure("correlation_matrix")

# %% [markdown]
# Finally, visualize the post-fit model and data.
# We first create the post-fit model prediction, using the model and the best-fit resuts.
#
# The visualization is using information stored in the workspace, which does not include binning or which observable is used.
# This information can be passed in via the `config` kwarg, but we can also edit the figure after its creation.
# We will demonstrate both approaches below.

# %%
# create post-fit model prediction
postfit_model = cabinetry.model_utils.prediction(model, fit_results=fit_results)

# binning to use in plot
plot_config = {
    "Regions": [
        {
            "Name": "Signal_region",
            "Binning": list(np.linspace(bin_edge_low, bin_edge_high, num_bins + 1)),
        }
    ]
}

figure_dict = cabinetry.visualize.data_mc(
    postfit_model, data, config=plot_config, save_figure=False
)

# modify x-axis label
fig = figure_dict[0]["figure"]
fig.axes[1].set_xlabel("m4l [GeV]")

# let's also save the figure
utils.save_figure("Signal_region_postfit")

# %% [markdown]
# We can also use `pyhf` directly. We already have a model and data, so let's calculate the CLs value.

# %%
mu_test = 1.0
cls = float(pyhf.infer.hypotest(mu_test, data, model))
print(f"CL_S for Signal_norm={mu_test} is {cls:.3f}")

# %% [markdown]
# ## Final remarks: control flow and duplication of information
#
# In the demonstration today we first built histograms, making sure we produce all that we will need for our statistical model.
# We then built the model, saying where to find each histogram.
# If we want to add a new histogram-based systematic uncertainty, we will need to implement it in two places:
# - in the histogram-producing code (e.g. coffea processor),
# - in the model building instructions.
#
# It can be convenient to avoid this duplication, which can be achieved as follows:
# - specify all relevant information in the model building configuration,
# - use that information to steer the histogram production.
#
# This method is available in `cabinetry`, check out the [cabinetry tutorials repository](https://github.com/cabinetry/cabinetry-tutorials) to see it in action.
# With this approach `cabinetry` builds the histograms it needs itself.
#
# This works well for some cases, but not for others:
# - a simple cut can be specified easily in the configuration, and varied for systematic uncertainties,
# - a complex calculation cannot easily be captured in such a way.
#
# The image below describes different options.
# The general considerations here are independent of the exact tools used, and should apply equally when using similar workflows.
#
# ![schematic for control flow options](utils/control_flow.png)
#
# A flexible design that works well for all scenarios is needed here, but is not available yet.
# If you have thoughts about this, please do get in touch!

# %% [markdown]
# ## Exercises
#
# Here are a few ideas to try out to become more familiar with the tools shown in this notebook:
#
# - Run the notebook a second time. It should be faster now — you are taking advantage of caching!
# - Change the event pre-selection in the `func_adl` query. Try out requiring exactly zero jets. Has the analysis become more or less sensitive after this change?
# - Change the lepton requirements. What happens when only accepting events with 4 electrons or 4 muons?
# - Try a different binning. Note how you still benefit from caching in the `ServiceX`-delivered data!
# - Compare the pre- and post-fit data/MC agreement (hint: the `fit_results` kwarg in `cabinetry.model_utils.prediction` is optional).
# - Find the value of the `Signal_norm` normalization factor for which CL_S = 0.05, the 95% confidence level upper parameter limit (hint: use `pyhf` directly, or `cabinetry.fit.limit`).
# - Separate the $Z$ and $t\bar{t}$ backgrounds and add a 6% normalization uncertainty for $t\bar{t}$ in the fit.
# - Replace the 10% normalization uncertainty for the $ZZ^\star$ background by a free-floating normalization factor in the fit.
#
#
# Advanced ideas:
# - Implement a more realistic systematic uncertainty in the `coffea` processor, for example for detector-related uncertainties for lepton kinematics. Propagate it through the analysis chain to observe the impact in a ranking plot produced with `cabinetry`.
# - Try out this workflow with your own analysis! Are there missing features or ways to streamline the experience? If so, please let us know!
