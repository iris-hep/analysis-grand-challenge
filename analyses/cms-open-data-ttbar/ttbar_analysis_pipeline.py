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
# # CMS Open Data $t\bar{t}$: from data delivery to statistical inference
#
# We are using [2015 CMS Open Data](https://cms.cern/news/first-cms-open-data-lhc-run-2-released) in this demonstration to showcase an analysis pipeline.
# It features data delivery and processing, histogram construction and visualization, as well as statistical inference.
#
# This notebook was developed in the context of the [IRIS-HEP AGC tools 2022 workshop](https://indico.cern.ch/e/agc-tools-2).
# This work was supported by the U.S. National Science Foundation (NSF) Cooperative Agreement OAC-1836650 (IRIS-HEP).
#
# This is a **technical demonstration**.
# We are including the relevant workflow aspects that physicists need in their work, but we are not focusing on making every piece of the demonstration physically meaningful.
# This concerns in particular systematic uncertainties: we capture the workflow, but the actual implementations are more complex in practice.
# If you are interested in the physics side of analyzing top pair production, check out the latest results from [ATLAS](https://twiki.cern.ch/twiki/bin/view/AtlasPublic/TopPublicResults) and [CMS](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/)!
# If you would like to see more technical demonstrations, also check out an [ATLAS Open Data example](https://indico.cern.ch/event/1076231/contributions/4560405/) demonstrated previously.
#
# This notebook implements most of the analysis pipeline shown in the following picture, using the tools also mentioned there:
# ![ecosystem visualization](utils/ecosystem.png)

# %% [markdown]
# ### Data pipelines
#
# There are two possible pipelines: one with `ServiceX` enabled, and one using only `coffea` for processing.
# ![processing pipelines](utils/processing_pipelines.png)

# %% [markdown]
# ### Imports: setting up our environment

# %% tags=[]
import asyncio
import logging
import os
import time

import awkward as ak
import cabinetry
import correctionlib
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from func_adl import ObjectStream
from func_adl_servicex import ServiceXSourceUpROOT
import hist
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
import uproot
from xgboost import XGBClassifier
import pyhf

import utils  # contains code for bookkeeping and cosmetics, as well as some boilerplate

logging.getLogger("cabinetry").setLevel(logging.INFO)

# %% [markdown]
# ### Configuration: number of files and data delivery path
#
# The number of files per sample set here determines the size of the dataset we are processing. There are 9 samples being used here, all part of the 2015 CMS Open Data release.
#
# These samples were originally published in miniAOD format, but for the purposes of this demonstration were pre-converted into nanoAOD format. More details about the inputs can be found [here](https://github.com/iris-hep/analysis-grand-challenge/tree/main/datasets/cms-open-data-2015).
#
# The table below summarizes the amount of data processed depending on the `N_FILES_MAX_PER_SAMPLE` setting.
#
# | setting | number of files | total size | number of events |
# | --- | --- | --- | --- |
# | `1` | 9 | 22.9 GB | 10455719 |
# | `2` | 18 | 42.8 GB | 19497435 |
# | `5` | 43 | 105 GB | 47996231 |
# | `10` | 79 | 200 GB | 90546458 |
# | `20` | 140 | 359 GB | 163123242 |
# | `50` | 255 | 631 GB | 297247463 |
# | `100` | 395 | 960 GB | 470397795 |
# | `200` | 595 | 1.40 TB | 705273291 |
# | `-1` | 787 | 1.78 TB | 940160174 |
#
# The input files are all in the 1–3 GB range.

# %% tags=[]
### GLOBAL CONFIGURATION
# input files per process, set to e.g. 10 (smaller number = faster)
N_FILES_MAX_PER_SAMPLE = 5

# enable Dask
USE_DASK = True

# enable ServiceX
USE_SERVICEX = False

### LOAD OTHER CONFIGURATION VARIABLES
with open("config.yaml") as config_file:
    config = yaml.safe_load(config_file)

### ML-INFERENCE SETTINGS

# enable ML inference
USE_INFERENCE = True

# enable inference using NVIDIA Triton server
USE_TRITON = False

### LOAD OTHER CONFIGURATION VARIABLES
with open("config.yaml") as config_file:
    config = yaml.safe_load(config_file)

config["ml"]["USE_INFERENCE"] = USE_INFERENCE
config["ml"]["USE_TRITON"] = USE_TRITON

# %% [markdown]
# ### Machine Learning Task
#
# During the processing step, machine learning is used to calculate one of the variables used for this analysis. The models used are trained separately in the `jetassignment_training.ipynb` notebook. Jets in the events are assigned to labels corresponding with their parent partons using a boosted decision tree (BDT). More information about the model and training can be found within that notebook. To obtain the features used as inputs for the BDT, we use the methods defined below:

# %% tags=[]
permutations_dict = utils.get_permutations_dict(config["ml"]["MAX_N_JETS"])


# %% tags=[]
def get_features(jets, electrons, muons, permutations_dict):
    '''
    Calculate features for each of the 12 combinations per event

    Args:
        jets: selected jets
        electrons: selected electrons
        muons: selected muons
        permutations_dict: which permutations to consider for each number of jets in an event

    Returns:
        features (flattened to remove event level)
        perm_counts: how many permutations in each event. use to unflatten features
    '''

    # calculate number of jets in each event
    njet = ak.num(jets).to_numpy()
    # don't consider every jet for events with high jet multiplicity
    njet[njet>max(permutations_dict.keys())] = max(permutations_dict.keys())
    # create awkward array of permutation indices
    perms = ak.Array([permutations_dict[n] for n in njet])
    perm_counts = ak.num(perms)

    #### calculate features ####
    features = np.zeros((sum(perm_counts),20))

    # grab lepton info
    leptons = ak.flatten(ak.concatenate((electrons, muons),axis=1),axis=-1)

    feature_count = 0

    # delta R between b_toplep and lepton
    features[:,0] = ak.flatten(np.sqrt((leptons.eta - jets[perms[...,3]].eta)**2 +
                                       (leptons.phi - jets[perms[...,3]].phi)**2)).to_numpy()


    #delta R between the two W
    features[:,1] = ak.flatten(np.sqrt((jets[perms[...,0]].eta - jets[perms[...,1]].eta)**2 +
                                       (jets[perms[...,0]].phi - jets[perms[...,1]].phi)**2)).to_numpy()

    #delta R between W and b_tophad
    features[:,2] = ak.flatten(np.sqrt((jets[perms[...,0]].eta - jets[perms[...,2]].eta)**2 +
                                       (jets[perms[...,0]].phi - jets[perms[...,2]].phi)**2)).to_numpy()
    features[:,3] = ak.flatten(np.sqrt((jets[perms[...,1]].eta - jets[perms[...,2]].eta)**2 +
                                       (jets[perms[...,1]].phi - jets[perms[...,2]].phi)**2)).to_numpy()

    # combined mass of b_toplep and lepton
    features[:,4] = ak.flatten((leptons + jets[perms[...,3]]).mass).to_numpy()

    # combined mass of W
    features[:,5] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]]).mass).to_numpy()

    # combined mass of W and b_tophad
    features[:,6] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]] +
                                 jets[perms[...,2]]).mass).to_numpy()

    feature_count+=1
    # combined pT of W and b_tophad
    features[:,7] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]] +
                                 jets[perms[...,2]]).pt).to_numpy()


    # pt of every jet
    features[:,8] = ak.flatten(jets[perms[...,0]].pt).to_numpy()
    features[:,9] = ak.flatten(jets[perms[...,1]].pt).to_numpy()
    features[:,10] = ak.flatten(jets[perms[...,2]].pt).to_numpy()
    features[:,11] = ak.flatten(jets[perms[...,3]].pt).to_numpy()

    # btagCSVV2 of every jet
    features[:,12] = ak.flatten(jets[perms[...,0]].btagCSVV2).to_numpy()
    features[:,13] = ak.flatten(jets[perms[...,1]].btagCSVV2).to_numpy()
    features[:,14] = ak.flatten(jets[perms[...,2]].btagCSVV2).to_numpy()
    features[:,15] = ak.flatten(jets[perms[...,3]].btagCSVV2).to_numpy()

    # quark-gluon likelihood discriminator of every jet
    features[:,16] = ak.flatten(jets[perms[...,0]].qgl).to_numpy()
    features[:,17] = ak.flatten(jets[perms[...,1]].qgl).to_numpy()
    features[:,18] = ak.flatten(jets[perms[...,2]].qgl).to_numpy()
    features[:,19] = ak.flatten(jets[perms[...,3]].qgl).to_numpy()

    return features, perm_counts


# %% [markdown]
# ### Defining our `coffea` Processor
#
# The processor includes a lot of the physics analysis details:
# - event filtering and the calculation of observables,
# - event weighting,
# - calculating systematic uncertainties at the event and object level,
# - filling all the information into histograms that get aggregated and ultimately returned to us by `coffea`.

# %% tags=[]
# functions creating systematic variations
def jet_pt_resolution(pt):
    # normal distribution with 5% variations, shape matches jets
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
    return ak.unflatten(resolution_variation, counts)

class TtbarAnalysis(processor.ProcessorABC):
    def __init__(self,
                 use_dask,
                 disable_processing,
                 io_branches,
                 ml_options,
                 xgboost_model_even,
                 xgboost_model_odd,
                 permutations_dict={}):

        num_bins = 25
        bin_low = 50
        bin_high = 550
        name = "observable"
        label = "observable [GeV]"
        self.hist = (
            hist.Hist.new.Reg(num_bins, bin_low, bin_high, name=name, label=label)
            .StrCat(["4j1b", "4j2b"], name="region", label="Region")
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )

        self.use_dask = use_dask
        self.disable_processing = disable_processing
        self.io_branches = io_branches
        self.cset = correctionlib.CorrectionSet.from_file("corrections.json")

        self.use_inference = ml_options["USE_INFERENCE"]
        if self.use_inference:
            self.ml_hist_dict = {}
            self.feature_names = ml_options["FEATURE_NAMES"]
            feature_descriptions = ml_options["FEATURE_DESCRIPTIONS"]
            for i in range(len(self.feature_names)):
                self.ml_hist_dict[f"hist_{self.feature_names[i]}"] = (
                    hist.Hist.new.Reg(num_bins,
                                      ml_options["BIN_RANGES"][i][0],
                                      ml_options["BIN_RANGES"][i][1],
                                      name="observable",
                                      label=feature_descriptions[i])
                    .StrCat([], name="process", label="Process", growth=True)
                    .StrCat([], name="variation", label="Systematic variation", growth=True)
                    .Weight()
                )

            # for ML inference
            self.use_triton = ml_options["USE_TRITON"]
            self.xgboost_model_even = xgboost_model_even
            self.xgboost_model_odd = xgboost_model_odd
            self.model_name = ml_options["MODEL_NAME"]
            self.model_vers_even = ml_options["MODEL_VERSION_EVEN"]
            self.model_vers_odd = ml_options["MODEL_VERSION_ODD"]
            self.url = ml_options["TRITON_URL"]
            self.permutations_dict = permutations_dict

    def only_do_IO(self, events):
        for branch in self.io_branches:
            if "_" in branch:
                split = branch.split("_")
                object_type = split[0]
                property_name = '_'.join(split[1:])
                ak.materialized(events[object_type][property_name])
            else:
                ak.materialized(events[branch])
        return {"hist": {}}

    def process(self, events):
        if self.disable_processing:
            # IO testing with no subsequent processing
            return self.only_do_IO(events)

        histogram = self.hist.copy()
        if self.use_inference:
            ml_hist_dict = {}
            for i in range(len(self.feature_names)):
                ml_hist_dict[f"hist_{self.feature_names[i]}"] = self.ml_hist_dict[f"hist_{self.feature_names[i]}"].copy()

        process = events.metadata["process"]  # "ttbar" etc.
        variation = events.metadata["variation"]  # "nominal" etc.

        # normalization for MC
        x_sec = events.metadata["xsec"]
        nevts_total = events.metadata["nevts"]
        lumi = 3378 # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / nevts_total
        else:
            xsec_weight = 1

        # setup triton gRPC client
        if self.use_inference:
            if self.use_triton:
                import tritonclient.grpc as grpcclient
                triton_client = grpcclient.InferenceServerClient(url=self.url)
                model_metadata = triton_client.get_model_metadata(self.model_name, self.model_vers_even)
                input_name = model_metadata.inputs[0].name
                dtype = model_metadata.inputs[0].datatype
                output_name = model_metadata.outputs[0].name

            elif not self.use_dask:
                model_even = XGBClassifier()
                model_even.load_model(self.xgboost_model_even)
                self.xgboost_model_even = model_even

                model_odd = XGBClassifier()
                model_odd.load_model(self.xgboost_model_odd)
                self.xgboost_model_odd = model_odd

        #### systematics
        # jet energy scale / resolution systematics
        # need to adjust schema to instead use coffea add_systematic feature, especially for ServiceX
        # cannot attach pT variations to events.jet, so attach to events directly
        # and subsequently scale pT by these scale factors
        events["pt_scale_up"] = 1.03
        events["pt_res_up"] = jet_pt_resolution(events.Jet.pt)

        syst_variations = ["nominal"]
        jet_kinematic_systs = ["pt_scale_up", "pt_res_up"]
        event_systs = [f"btag_var_{i}" for i in range(4)]
        if process == "wjets":
            event_systs.append("scale_var")

        # Only do systematics for nominal samples, e.g. ttbar__nominal
        if variation == "nominal":
            syst_variations.extend(jet_kinematic_systs)
            syst_variations.extend(event_systs)

        # for pt_var in pt_variations:
        for syst_var in syst_variations:
            ### event selection
            # very very loosely based on https://arxiv.org/abs/2006.13076

            # Note: This creates new objects, distinct from those in the 'events' object
            elecs = events.Electron
            muons = events.Muon
            jets = events.Jet
            if syst_var in jet_kinematic_systs:
                # Replace jet.pt with the adjusted values
                jets["pt"] = jets.pt * events[syst_var]

            electron_reqs = (elecs.pt > 30) & (np.abs(elecs.eta) < 2.1) & (elecs.cutBased == 4) & (elecs.sip3d < 4)
            muon_reqs = ((muons.pt > 30) & (np.abs(muons.eta) < 2.1) & (muons.tightId) & (muons.sip3d < 4) &
                         (muons.pfRelIso04_all < 0.15))
            jet_reqs = (jets.pt > 30) & (np.abs(jets.eta) < 2.4) & (jets.isTightLeptonVeto)

            # Only keep objects that pass our requirements
            elecs = elecs[electron_reqs]
            muons = muons[muon_reqs]
            jets = jets[jet_reqs]

            if self.use_inference:
                even = (events.event%2==0)  # whether events are even/odd

            B_TAG_THRESHOLD = 0.5

            ######### Store boolean masks with PackedSelection ##########
            selections = PackedSelection(dtype='uint64')
            # Basic selection criteria
            selections.add("exactly_1l", (ak.num(elecs) + ak.num(muons)) == 1)
            selections.add("atleast_4j", ak.num(jets) >= 4)
            selections.add("exactly_1b", ak.sum(jets.btagCSVV2 >= B_TAG_THRESHOLD, axis=1) == 1)
            selections.add("atleast_2b", ak.sum(jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2)
            # Complex selection criteria
            selections.add("4j1b", selections.all("exactly_1l", "atleast_4j", "exactly_1b"))
            selections.add("4j2b", selections.all("exactly_1l", "atleast_4j", "atleast_2b"))

            for region in ["4j1b", "4j2b"]:
                region_selection = selections.all(region)
                region_jets = jets[region_selection]
                region_elecs = elecs[region_selection]
                region_muons = muons[region_selection]
                region_weights = np.ones(len(region_jets)) * xsec_weight
                if self.use_inference:
                    region_even = even[region_selection]

                if region == "4j1b":
                    observable = ak.sum(region_jets.pt, axis=-1)

                elif region == "4j2b":

                    # reconstruct hadronic top as bjj system with largest pT
                    trijet = ak.combinations(region_jets, 3, fields=["j1", "j2", "j3"])  # trijet candidates
                    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
                    trijet["max_btag"] = np.maximum(trijet.j1.btagCSVV2, np.maximum(trijet.j2.btagCSVV2, trijet.j3.btagCSVV2))
                    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # at least one-btag in trijet candidates
                    # pick trijet candidate with largest pT and calculate mass of system
                    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
                    observable = ak.flatten(trijet_mass)

                    if sum(region_selection)==0: continue

                    if self.use_inference:
                        features, perm_counts = get_features(region_jets, region_elecs, region_muons, self.permutations_dict)
                        even_perm = np.repeat(region_even, perm_counts)

                        #calculate ml observable
                        if self.use_triton:

                            results = np.zeros(features.shape[0])
                            output = grpcclient.InferRequestedOutput(output_name)

                            if len(features[even_perm])>0:
                                inpt = [grpcclient.InferInput(input_name, features[even_perm].shape, dtype)]
                                inpt[0].set_data_from_numpy(features[even_perm].astype(np.float32))
                                results[even_perm]=triton_client.infer(
                                    model_name=self.model_name,
                                    model_version=self.model_vers_even,
                                    inputs=inpt,
                                    outputs=[output]
                                ).as_numpy(output_name)[:, 1]
                            if len(features[np.invert(even_perm)])>0:
                                inpt = [grpcclient.InferInput(input_name, features[np.invert(even_perm)].shape, dtype)]
                                inpt[0].set_data_from_numpy(features[np.invert(even_perm)].astype(np.float32))
                                results[np.invert(even_perm)]=triton_client.infer(
                                    model_name=self.model_name,
                                    model_version=self.model_vers_odd,
                                    inputs=inpt,
                                    outputs=[output]
                                ).as_numpy(output_name)[:, 1]

                        else:
                            results = np.zeros(features.shape[0])
                            if len(features[even_perm])>0:
                                results[even_perm] = self.xgboost_model_odd.predict_proba(
                                    features[even_perm,:])[:, 1]
                            if len(features[np.invert(even_perm)])>0:
                                results[np.invert(even_perm)] = results_odd = self.xgboost_model_even.predict_proba(
                                    features[np.invert(even_perm),:])[:, 1]

                        results = ak.unflatten(results, perm_counts)
                        features = ak.flatten(ak.unflatten(features, perm_counts)[
                            ak.from_regular(ak.argmax(results,axis=1)[:, np.newaxis])
                        ])
                syst_var_name = f"{syst_var}"
                # Break up the filling into event weight systematics and object variation systematics
                if syst_var in event_systs:
                    for i_dir, direction in enumerate(["up", "down"]):
                        # Should be an event weight systematic with an up/down variation
                        if syst_var.startswith("btag_var"):
                            i_jet = int(syst_var.rsplit("_",1)[-1])   # Kind of fragile
                            wgt_variation = self.cset["event_systematics"].evaluate("btag_var", direction, region_jets.pt[:,i_jet])
                        elif syst_var == "scale_var":
                            # The pt array is only used to make sure the output array has the correct shape
                            wgt_variation = self.cset["event_systematics"].evaluate("scale_var", direction, region_jets.pt[:,0])
                        syst_var_name = f"{syst_var}_{direction}"
                        histogram.fill(
                            observable=observable, region=region, process=process,
                            variation=syst_var_name, weight=region_weights * wgt_variation
                        )
                        if region=="4j2b" and self.use_inference:
                            for i in range(len(self.feature_names)):
                                ml_hist_dict[f"hist_{self.feature_names[i]}"].fill(observable=features[...,i],
                                                                                   process=process,
                                                                                   variation=syst_var_name,
                                                                                   weight=region_weights * wgt_variation)
                else:
                    # Should either be 'nominal' or an object variation systematic
                    if variation != "nominal":
                        # This is a 2-point systematic, e.g. ttbar__scaledown, ttbar__ME_var, etc.
                        syst_var_name = variation
                    histogram.fill(
                        observable=observable, region=region, process=process,
                        variation=syst_var_name, weight=region_weights
                    )
                    if region=="4j2b" and self.use_inference:
                        for i in range(len(self.feature_names)):
                            ml_hist_dict[f"hist_{self.feature_names[i]}"].fill(observable=features[...,i],
                                                                               process=process,
                                                                               variation=syst_var_name,
                                                                               weight=region_weights)


        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist": histogram}
        if self.use_inference:
            output["ml_hist_dict"] = ml_hist_dict

        return output

    def postprocess(self, accumulator):
        return accumulator

# %% [markdown]
# ### "Fileset" construction and metadata
#
# Here, we gather all the required information about the files we want to process: paths to the files and asociated metadata.

# %% tags=[]
fileset = utils.construct_fileset(N_FILES_MAX_PER_SAMPLE, use_xcache=False, af_name=config["benchmarking"]["AF_NAME"])  # local files on /data for ssl-dev

print(f"processes in fileset: {list(fileset.keys())}")
print(f"\nexample of information in fileset:\n{{\n  'files': [{fileset['ttbar__nominal']['files'][0]}, ...],")
print(f"  'metadata': {fileset['ttbar__nominal']['metadata']}\n}}")


# %% [markdown]
# ### ServiceX-specific functionality: query setup
#
# Define the func_adl query to be used for the purpose of extracting columns and filtering.

# %% tags=[]
def get_query(source: ObjectStream) -> ObjectStream:
    """Query for event / column selection: >=4j >=1b, ==1 lep with pT>30 GeV + additional cuts, return relevant columns
    """
    cuts = source.Where(lambda e: {"pt": e.Electron_pt, 
                               "eta": e.Electron_eta, 
                               "cutBased": e.Electron_cutBased, 
                               "sip3d": e.Electron_sip3d,}.Zip()\
                        .Where(lambda electron: (electron.pt > 30
                                                 and abs(electron.eta) < 2.1 
                                                 and electron.cutBased == 4
                                                 and electron.sip3d < 4)).Count() 
                        + {"pt": e.Muon_pt, 
                           "eta": e.Muon_eta,
                           "tightId": e.Muon_tightId,
                           "sip3d": e.Muon_sip3d,
                           "pfRelIso04_all": e.Muon_pfRelIso04_all}.Zip()\
                        .Where(lambda muon: (muon.pt > 30 
                                             and abs(muon.eta) < 2.1 
                                             and muon.tightId 
                                             and muon.pfRelIso04_all < 0.15)).Count()== 1)\
                        .Where(lambda f: {"pt": f.Jet_pt, 
                                          "eta": f.Jet_eta,
                                          "jetId": f.Jet_jetId}.Zip()\
                               .Where(lambda jet: (jet.pt > 30 
                                                   and abs(jet.eta) < 2.4 
                                                   and jet.jetId == 6)).Count() >= 4)\
                        .Where(lambda g: {"pt": g.Jet_pt, 
                                          "eta": g.Jet_eta,
                                          "btagCSVV2": g.Jet_btagCSVV2,
                                          "jetId": g.Jet_jetId}.Zip()\
                        .Where(lambda jet: (jet.btagCSVV2 >= 0.5 
                                            and jet.pt > 30
                                            and abs(jet.eta) < 2.4) 
                                            and jet.jetId == 6).Count() >= 1)
    selection = cuts.Select(lambda h: {"Electron_pt": h.Electron_pt,
                                       "Electron_eta": h.Electron_eta,
                                       "Electron_phi": h.Electron_phi,
                                       "Electron_mass": h.Electron_mass,
                                       "Electron_cutBased": h.Electron_cutBased,
                                       "Electron_sip3d": h.Electron_sip3d,
                                       "Muon_pt": h.Muon_pt,
                                       "Muon_eta": h.Muon_eta,
                                       "Muon_phi": h.Muon_phi,
                                       "Muon_mass": h.Muon_mass,
                                       "Muon_tightId": h.Muon_tightId,
                                       "Muon_sip3d": h.Muon_sip3d,
                                       "Muon_pfRelIso04_all": h.Muon_pfRelIso04_all,
                                       "Jet_mass": h.Jet_mass,
                                       "Jet_pt": h.Jet_pt,
                                       "Jet_eta": h.Jet_eta,
                                       "Jet_phi": h.Jet_phi,
                                       "Jet_qgl": h.Jet_qgl,
                                       "Jet_btagCSVV2": h.Jet_btagCSVV2,
                                       "Jet_jetId": h.Jet_jetId,
                                       "event": h.event,
                                      })
    if USE_INFERENCE:
        return selection
    
    # some branches are only needed if USE_INFERENCE is turned on
    return selection.Select(lambda h: {"Electron_pt": h.Electron_pt,
                                       "Electron_eta": h.Electron_eta,
                                       "Electron_cutBased": h.Electron_cutBased,
                                       "Electron_sip3d": h.Electron_sip3d,
                                       "Muon_pt": h.Muon_pt,
                                       "Muon_eta": h.Muon_eta,
                                       "Muon_tightId": h.Muon_tightId,
                                       "Muon_sip3d": h.Muon_sip3d,
                                       "Muon_pfRelIso04_all": h.Muon_pfRelIso04_all,
                                       "Jet_mass": h.Jet_mass,
                                       "Jet_pt": h.Jet_pt,
                                       "Jet_eta": h.Jet_eta,
                                       "Jet_phi": h.Jet_phi,
                                       "Jet_btagCSVV2": h.Jet_btagCSVV2,
                                       "Jet_jetId": h.Jet_jetId,
                                      })


# %% [markdown]
# ### Caching the queried datasets with `ServiceX`
#
# Using the queries created with `func_adl`, we are using `ServiceX` to read the CMS Open Data files to build cached files with only the specific event information as dictated by the query.

# %% tags=[]
if USE_SERVICEX:
    # dummy dataset on which to generate the query
    dummy_ds = ServiceXSourceUpROOT("cernopendata://dummy", "Events", backend_name="uproot")

    # tell low-level infrastructure not to contact ServiceX yet, only to
    # return the qastle string it would have sent
    dummy_ds.return_qastle = True

    # create the query
    query = get_query(dummy_ds).value()

    # now we query the files using a wrapper around ServiceXDataset to transform all processes at once
    t0 = time.time()
    ds = utils.ServiceXDatasetGroup(fileset, backend_name="uproot", ignore_cache=config["global"]["SERVICEX_IGNORE_CACHE"])
    files_per_process = ds.get_data_rootfiles_uri(query, as_signed_url=True, title="CMS ttbar")

    print(f"ServiceX data delivery took {time.time() - t0:.2f} seconds")

    # update fileset to point to ServiceX-transformed files
    for process in fileset.keys():
        fileset[process]["files"] = [f.url for f in files_per_process[process]]

# %% [markdown]
# ### Execute the data delivery pipeline
#
# What happens here depends on the flag `USE_SERVICEX`. If set to true, the processor is run on the data previously gathered by ServiceX, then will gather output histograms.
#
# When `USE_SERVICEX` is false, the input files need to be processed during this step as well.

# %% tags=[]
NanoAODSchema.warn_missing_crossrefs = False # silences warnings about branches we will not use here
if USE_DASK:
    executor = processor.DaskExecutor(client=utils.get_client(af=config["global"]["AF"]))
else:
    executor = processor.FuturesExecutor(workers=config["benchmarking"]["NUM_CORES"])

run = processor.Runner(executor=executor, schema=NanoAODSchema, savemetrics=True, metadata_cache={}, chunksize=config["benchmarking"]["CHUNKSIZE"])

if USE_SERVICEX:
    treename = "servicex"

else:
    treename = "Events"

if not USE_DASK and not USE_TRITON and USE_INFERENCE:
    model_even = config["ml"]["XGBOOST_MODEL_PATH_EVEN"]
    model_odd = config["ml"]["XGBOOST_MODEL_PATH_ODD"]

elif not USE_TRITON and USE_INFERENCE:
    model_even = XGBClassifier()
    model_even.load_model(config["ml"]["XGBOOST_MODEL_PATH_EVEN"])
    model_odd = XGBClassifier()
    model_odd.load_model(config["ml"]["XGBOOST_MODEL_PATH_ODD"])

else:
    model_even = None
    model_odd = None

filemeta = run.preprocess(fileset, treename=treename)  # pre-processing

t0 = time.monotonic()
all_histograms, metrics = run(fileset, treename, processor_instance=TtbarAnalysis(USE_DASK,
                                                                                  config["benchmarking"]["DISABLE_PROCESSING"],
                                                                                  config["benchmarking"]["IO_BRANCHES"][
                                                                                      config["benchmarking"]["IO_FILE_PERCENT"]
                                                                                  ],
                                                                                  config["ml"],
                                                                                  model_even,
                                                                                  model_odd,
                                                                                  permutations_dict))  # processing
exec_time = time.monotonic() - t0

print(f"\nexecution took {exec_time:.2f} seconds")

# %%
# track metrics
dataset_source = "/data" if fileset["ttbar__nominal"]["files"][0].startswith("/data") else "https://xrootd-local.unl.edu:1094" # TODO: xcache support
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
    "io_file_percent": config["benchmarking"]["IO_FILE_PERCENT"]
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
print(f"amount of data read: {metrics['bytesread']/1000**2:.2f} MB")  # likely buggy: https://github.com/CoffeaTeam/coffea/issues/717

# %% [markdown]
# ### Inspecting the produced histograms
#
# Let's have a look at the data we obtained.
# We built histograms in two phase space regions, for multiple physics processes and systematic variations.

# %% tags=[]
utils.set_style()

all_histograms["hist"][120j::hist.rebin(2), "4j1b", :, "nominal"].stack("process")[::-1].plot(stack=True, histtype="fill", linewidth=1, edgecolor="grey")
plt.legend(frameon=False)
plt.title("$\geq$ 4 jets, 1 b-tag")
plt.xlabel("$H_T$ [GeV]");

# %% tags=[]
all_histograms["hist"][:, "4j2b", :, "nominal"].stack("process")[::-1].plot(stack=True, histtype="fill", linewidth=1,edgecolor="grey")
plt.legend(frameon=False)
plt.title("$\geq$ 4 jets, $\geq$ 2 b-tags")
plt.xlabel("$m_{bjj}$ [GeV]");

# %% [markdown]
# Our top reconstruction approach ($bjj$ system with largest $p_T$) has worked!
#
# Let's also have a look at some systematic variations:
# - b-tagging, which we implemented as jet-kinematic dependent event weights,
# - jet energy variations, which vary jet kinematics, resulting in acceptance effects and observable changes.
#
# We are making of [UHI](https://uhi.readthedocs.io/) here to re-bin.

# %% tags=[]
# b-tagging variations
all_histograms["hist"][120j::hist.rebin(2), "4j1b", "ttbar", "nominal"].plot(label="nominal", linewidth=2)
all_histograms["hist"][120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_0_up"].plot(label="NP 1", linewidth=2)
all_histograms["hist"][120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_1_up"].plot(label="NP 2", linewidth=2)
all_histograms["hist"][120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_2_up"].plot(label="NP 3", linewidth=2)
all_histograms["hist"][120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_3_up"].plot(label="NP 4", linewidth=2)
plt.legend(frameon=False)
plt.xlabel("$H_T$ [GeV]")
plt.title("b-tagging variations");

# %% tags=[]
# jet energy scale variations
all_histograms["hist"][:, "4j2b", "ttbar", "nominal"].plot(label="nominal", linewidth=2)
all_histograms["hist"][:, "4j2b", "ttbar", "pt_scale_up"].plot(label="scale up", linewidth=2)
all_histograms["hist"][:, "4j2b", "ttbar", "pt_res_up"].plot(label="resolution up", linewidth=2)
plt.legend(frameon=False)
plt.xlabel("$m_{bjj}$ [Gev]")
plt.title("Jet energy variations");

# %% tags=[]
# ML inference variables
if USE_INFERENCE:
    fig, axs = plt.subplots(10,2,figsize=(14,40))
    for i in range(len(config["ml"]["FEATURE_NAMES"])):
        if i<10: 
            column=0
            row=i
        else: 
            column=1
            row=i-10
        all_histograms['ml_hist_dict'][f'hist_{config["ml"]["FEATURE_NAMES"][i]}'][:, :, "nominal"].stack("process").project("observable").plot(
            stack=True, 
            histtype="fill", 
            linewidth=1, 
            edgecolor="grey", 
            ax=axs[row,column])
        axs[row, column].legend(frameon=False)
    fig.show()

# %% [markdown]
# ### Save histograms to disk
#
# We'll save everything to disk for subsequent usage.
# This also builds pseudo-data by combining events from the various simulation setups we have processed.

# %% tags=[]
utils.save_histograms(all_histograms['hist'], fileset, "histograms.root")
if USE_INFERENCE:
    utils.save_ml_histograms(all_histograms['ml_hist_dict'], fileset, "histograms_ml.root", config)

# %% [markdown]
# ### Statistical inference
#
# A statistical model has been defined in `config.yml`, ready to be used with our output.
# We will use `cabinetry` to combine all histograms into a `pyhf` workspace and fit the resulting statistical model to the pseudodata we built.

# %% tags=[]
config = cabinetry.configuration.load("cabinetry_config.yml")
cabinetry.templates.collect(config)
cabinetry.templates.postprocess(config)  # optional post-processing (e.g. smoothing)
ws = cabinetry.workspace.build(config)
cabinetry.workspace.save(ws, "workspace.json")

# %% [markdown]
# We can inspect the workspace with `pyhf`, or use `pyhf` to perform inference.

# %% tags=[]
# !pyhf inspect workspace.json | head -n 20

# %% [markdown]
# Let's try out what we built: the next cell will perform a maximum likelihood fit of our statistical model to the pseudodata we built.

# %% tags=[]
model, data = cabinetry.model_utils.model_and_data(ws)
fit_results = cabinetry.fit.fit(model, data)

cabinetry.visualize.pulls(
    fit_results, exclude="ttbar_norm", close_figure=True, save_figure=False
)

# %% [markdown]
# For this pseudodata, what is the resulting ttbar cross-section divided by the Standard Model prediction?

# %% tags=[]
poi_index = model.config.poi_index
print(f"\nfit result for ttbar_norm: {fit_results.bestfit[poi_index]:.3f} +/- {fit_results.uncertainty[poi_index]:.3f}")

# %% [markdown]
# Let's also visualize the model before and after the fit, in both the regions we are using.
# The binning here corresponds to the binning used for the fit.

# %%
model_prediction = cabinetry.model_utils.prediction(model)
figs = cabinetry.visualize.data_mc(model_prediction, data, close_figure=True, config=config)
figs[0]["figure"]

# %% tags=[]
figs[1]["figure"]

# %% [markdown]
# We can see very good post-fit agreement.

# %% tags=[]
model_prediction_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
figs = cabinetry.visualize.data_mc(model_prediction_postfit, data, close_figure=True, config=config)
figs[0]["figure"]

# %% tags=[]
figs[1]["figure"]

# %% [markdown]
# ### ML Validation
# We can further validate our results by applying the above fit to different ML observables and checking for good agreement.

# %% tags=[]
# load the ml workspace (uses the ml observable instead of previous method)
if USE_INFERENCE:
    config_ml = cabinetry.configuration.load("cabinetry_config_ml.yml")
    cabinetry.templates.collect(config_ml)
    cabinetry.templates.postprocess(config_ml)  # optional post-processing (e.g. smoothing)

    ws_ml = cabinetry.workspace.build(config_ml)
    ws_pruned = pyhf.Workspace(ws_ml).prune(channels=["Feature3", "Feature8", "Feature9",
                                                      "Feature10", "Feature11", "Feature12",
                                                      "Feature13", "Feature14", "Feature15",
                                                      "Feature16", "Feature17", "Feature18",
                                                      "Feature19"])

    cabinetry.workspace.save(ws_pruned, "workspace_ml.json")

# %% tags=[]
if USE_INFERENCE:
    model_ml, data_ml = cabinetry.model_utils.model_and_data(ws_pruned)

# %% [markdown]
# We have a channel for each ML observable:

# %%
# !pyhf inspect workspace_ml.json | head -n 20

# %%
# obtain model prediction before and after fit
if USE_INFERENCE:
    model_prediction = cabinetry.model_utils.prediction(model_ml)
    fit_results_mod = cabinetry.model_utils.match_fit_results(model_ml, fit_results)
    model_prediction_postfit = cabinetry.model_utils.prediction(model_ml, fit_results=fit_results_mod)

# %% tags=[]
if USE_INFERENCE:
    figs = utils.plot_data_mc(model_prediction, model_prediction_postfit, data_ml, config_ml)

# %% [markdown]
# ### What is next?
#
# Our next goals for this pipeline demonstration are:
# - making this analysis even **more feature-complete**,
# - **addressing performance bottlenecks** revealed by this demonstrator,
# - **collaborating** with you!
#
# Please do not hesitate to get in touch if you would like to join the effort, or are interested in re-implementing (pieces of) the pipeline with different tools!
#
# Our mailing list is analysis-grand-challenge@iris-hep.org, sign up via the [Google group](https://groups.google.com/a/iris-hep.org/g/analysis-grand-challenge).
