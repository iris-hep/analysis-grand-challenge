# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import asyncio
import time
import logging
import click
import mlflow

import vector; vector.register_awkward()

import awkward as ak
import cabinetry
from coffea import processor
from coffea.processor import servicex
from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from func_adl import ObjectStream
import hist
import json
import matplotlib.pyplot as plt
import numpy as np
import uproot

import utils  # contains code for bookkeeping and cosmetics, as well as some boilerplate

logging.getLogger("cabinetry").setLevel(logging.INFO)


processor_base = servicex.Analysis

# functions creating systematic variations
def flat_variation(ones):
    # 0.1% weight variations
    return (1.0 + np.array([0.001, -0.001], dtype=np.float32)) * ones[:, None]


def btag_weight_variation(i_jet, jet_pt):
    # weight variation depending on i-th jet pT (10% as default value, multiplied by i-th jet pT / 50 GeV)
    return 1 + np.array([0.1, -0.1]) * (ak.singletons(jet_pt[:, i_jet]) / 50).to_numpy()


def jet_pt_resolution(pt):
    # normal distribution with 5% variations, shape matches jets
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
    return ak.unflatten(resolution_variation, counts)


class TtbarAnalysis(processor_base):
    def __init__(self, num_bins, bin_low, bin_high, pt_threshold):
        name = "observable"
        label = "observable [GeV]"
        self.pt_threshold = pt_threshold
        self.hist = (
            hist.Hist.new.Reg(num_bins, bin_low, bin_high, name=name, label=label)
            .StrCat(["4j1b", "4j2b"], name="region", label="Region")
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )

    def process(self, events):
        histogram = self.hist.copy()

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

        #### systematics
        # example of a simple flat weight variation, using the coffea nanoevents systematics feature
        if process == "wjets":
            events.add_systematic("scale_var", "UpDownSystematic", "weight", flat_variation)

        # jet energy scale / resolution systematics
        # need to adjust schema to instead use coffea add_systematic feature, especially for ServiceX
        # cannot attach pT variations to events.jet, so attach to events directly
        # and subsequently scale pT by these scale factors
        events["pt_nominal"] = 1.0
        events["pt_scale_up"] = 1.03
        events["pt_res_up"] = jet_pt_resolution(events.jet.pt)

        pt_variations = ["pt_nominal", "pt_scale_up", "pt_res_up"] if variation == "nominal" else ["pt_nominal"]
        for pt_var in pt_variations:

            ### event selection
            # very very loosely based on https://arxiv.org/abs/2006.13076

            # pT > 25 GeV for leptons & jets
            selected_electrons = events.electron[events.electron.pt > self.pt_threshold]
            selected_muons = events.muon[events.muon.pt > self.pt_threshold]
            jet_filter = events.jet.pt * events[pt_var] > self.pt_threshold
            selected_jets = events.jet[jet_filter]

            # single lepton requirement
            event_filters = (ak.count(selected_electrons.pt, axis=1) & ak.count(selected_muons.pt, axis=1) == 1)
            # at least four jets
            pt_var_modifier = events[pt_var] if "res" not in pt_var else events[pt_var][jet_filter]
            event_filters = event_filters & (ak.count(selected_jets.pt * pt_var_modifier, axis=1) >= 4)
            # at least one b-tagged jet ("tag" means score above threshold)
            B_TAG_THRESHOLD = 0.5
            event_filters = event_filters & (ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) >= 1)

            # apply event filters
            selected_events = events[event_filters]
            selected_electrons = selected_electrons[event_filters]
            selected_muons = selected_muons[event_filters]
            selected_jets = selected_jets[event_filters]

            for region in ["4j1b", "4j2b"]:
                # further filtering: 4j1b CR with single b-tag, 4j2b SR with two or more tags
                if region == "4j1b":
                    region_filter = ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) == 1
                    selected_jets_region = selected_jets[region_filter]
                    # use HT (scalar sum of jet pT) as observable
                    pt_var_modifier = events[event_filters][region_filter][pt_var] if "res" not in pt_var else events[pt_var][jet_filter][event_filters][region_filter]
                    observable = ak.sum(selected_jets_region.pt * pt_var_modifier, axis=-1)

                elif region == "4j2b":
                    region_filter = ak.sum(selected_jets.btag > B_TAG_THRESHOLD, axis=1) >= 2
                    selected_jets_region = selected_jets[region_filter]

                    # wrap into a four-vector object to allow addition
                    selected_jets_region = ak.zip(
                        {
                            "pt": selected_jets_region.pt, "eta": selected_jets_region.eta, "phi": selected_jets_region.phi,
                            "mass": selected_jets_region.mass, "btag": selected_jets_region.btag,
                        },
                        with_name="Momentum4D",
                    )

                    # reconstruct hadronic top as bjj system with largest pT
                    # the jet energy scale / resolution effect is not propagated to this observable at the moment
                    trijet = ak.combinations(selected_jets_region, 3, fields=["j1", "j2", "j3"])  # trijet candidates
                    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
                    trijet["max_btag"] = np.maximum(trijet.j1.btag, np.maximum(trijet.j2.btag, trijet.j3.btag))
                    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # require at least one-btag in trijet candidates
                    # pick trijet candidate with largest pT and calculate mass of system
                    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
                    observable = ak.flatten(trijet_mass)

                ### histogram filling
                if pt_var == "pt_nominal":
                    # nominal pT, but including 2-point systematics
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=variation, weight=xsec_weight
                        )

                    if variation == "nominal":
                        # also fill weight-based variations for all nominal samples
                        for weight_name in events.systematics.fields:
                            for direction in ["up", "down"]:
                                # extract the weight variations and apply all event & region filters
                                weight_variation = events.systematics[weight_name][direction][f"weight_{weight_name}"][event_filters][region_filter]
                                # fill histograms
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )

                        # calculate additional systematics: b-tagging variations
                        for i_var, weight_name in enumerate([f"btag_var_{i}" for i in range(4)]):
                            for i_dir, direction in enumerate(["up", "down"]):
                                # create systematic variations that depend on object properties (here: jet pT)
                                if len(observable):
                                    weight_variation = btag_weight_variation(i_var, selected_jets_region.pt)[:, 1-i_dir]
                                else:
                                    weight_variation = 1 # no events selected
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )

                elif variation == "nominal":
                    # pT variations for nominal samples
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=pt_var, weight=xsec_weight
                        )

        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist": histogram}

        return output

    def postprocess(self, accumulator):
        return accumulator


def get_query(source: ObjectStream) -> ObjectStream:
    """Query for event / column selection: no filter, select relevant lepton and jet columns
    """
    return source.Select(lambda e: {
                                    "electron_pt": e.electron_pt,
                                    "muon_pt": e.muon_pt,
                                    "jet_pt": e.jet_pt,
                                    "jet_eta": e.jet_eta,
                                    "jet_phi": e.jet_phi,
                                    "jet_mass": e.jet_mass,
                                    "jet_btag": e.jet_btag,
                                   }
                        )
@click.command(
    help="CMS Open Data t-tbar: from data delivery to statistical inference."
)
@click.option("--num-input-files", type=click.INT, default=10, help="input files per process, set to e.g. 10 (smaller number = faster).")
@click.option("--num-bins", type=click.INT, default=25, help="Number of bins.")
@click.option("--bin-low", type=click.INT, default=50, help="Bottom bin.")
@click.option("--bin-high", type=click.INT, default=550, help="Top bin.")
@click.option("--pt-threshold", type=click.INT, default=25, help="pt for leptons & jets (in GeV).")
def analysis(num_input_files, num_bins, bin_low, bin_high, pt_threshold):
    with mlflow.start_run():
        fileset = utils.construct_fileset(num_input_files, use_xcache=False)

        print(f"processes in fileset: {list(fileset.keys())}")
        print(
            f"\nexample of information in fileset:\n{{\n  'files': [{fileset['ttbar__nominal']['files'][0]}, ...],")
        print(f"  'metadata': {fileset['ttbar__nominal']['metadata']}\n}}")


        async def produce_all_the_histograms(fileset, analysis_processor):
            return await utils.produce_all_histograms(fileset, get_query, analysis_processor, use_dask=False)

        analysis_processor = TtbarAnalysis(num_bins, bin_low, bin_high, pt_threshold)
        all_histograms = asyncio.run(produce_all_the_histograms(fileset, analysis_processor))

        print(all_histograms)

        utils.set_style()

        all_histograms[120j::hist.rebin(2), "4j1b", :, "nominal"].stack("process")[::-1].plot(
            stack=True, histtype="fill", linewidth=1, edgecolor="grey")
        plt.legend(frameon=False)
        plt.title(">= 4 jets, 1 b-tag")
        plt.xlabel("HT [GeV]")
        plt.savefig('1-btag.png')
        plt.clf()
        mlflow.log_artifact("1-btag.png")

        all_histograms[:, "4j2b", :, "nominal"].stack("process")[::-1].plot(stack=True,
                                                                            histtype="fill",
                                                                            linewidth=1,
                                                                            edgecolor="grey")
        plt.legend(frameon=False)
        plt.title(">= 4 jets, >= 2 b-tags")
        plt.xlabel("$m_{bjj}$ [Gev]");
        plt.savefig('2-btag.png')
        plt.clf()

        mlflow.log_artifact("2-btag.png")

        # b-tagging variations
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "nominal"].plot(
            label="nominal", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_0_down"].plot(
            label="NP 1", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_1_down"].plot(
            label="NP 2", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_2_down"].plot(
            label="NP 3", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_3_down"].plot(
            label="NP 4", linewidth=2)
        plt.legend(frameon=False)
        plt.xlabel("HT [GeV]")
        plt.title("b-tagging variations");
        plt.savefig('b-tag-variations.png')
        plt.clf()

        mlflow.log_artifact("b-tag-variations.png")

        # jet energy scale variations
        all_histograms[:, "4j2b", "ttbar", "nominal"].plot(label="nominal", linewidth=2)
        all_histograms[:, "4j2b", "ttbar", "pt_scale_up"].plot(label="scale up", linewidth=2)
        all_histograms[:, "4j2b", "ttbar", "pt_res_up"].plot(label="resolution up", linewidth=2)
        plt.legend(frameon=False)
        plt.xlabel("$m_{bjj}$ [Gev]")
        plt.title("Jet energy variations")
        plt.savefig('jet-energy-variations.png')
        plt.clf()

        mlflow.log_artifact("jet-energy-variations.png")

        utils.save_histograms(all_histograms, fileset, "histograms.root")
        mlflow.log_artifact("histograms.root")

        config = cabinetry.configuration.load("cabinetry_config.yml")
        cabinetry.templates.collect(config)
        cabinetry.templates.postprocess(
            config)  # optional post-processing (e.g. smoothing)
        ws = cabinetry.workspace.build(config)
        cabinetry.workspace.save(ws, "workspace.json")

        model, data = cabinetry.model_utils.model_and_data(ws)
        fit_results = cabinetry.fit.fit(model, data)

        fig = cabinetry.visualize.pulls(
            fit_results, exclude="ttbar_norm", close_figure=True, save_figure=False
        )
        fig.savefig("ttbar_norm_fit.png")
        plt.clf()
        mlflow.log_artifact("ttbar_norm_fit.png")

        poi_index = model.config.poi_index
        mlflow.log_metric("ttbar_norm_bestfit", fit_results.bestfit[poi_index])
        mlflow.log_metric("ttbar_norm_uncertainty", fit_results.uncertainty[poi_index])

        print(
            f"\nfit result for ttbar_norm: {fit_results.bestfit[poi_index]:.3f} +/- {fit_results.uncertainty[poi_index]:.3f}")


if __name__ == "__main__":
    analysis()