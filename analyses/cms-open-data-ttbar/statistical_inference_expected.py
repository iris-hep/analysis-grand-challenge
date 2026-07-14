# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.9
# ---

# %% [markdown]
# # Statistical Inference with Combine - expected results
#
# When working on a real analysis, you do not start immediately running fits using the observed data. By doing so, you could indeed bias the design of your analysis.
# For this reason, it is important to produce the so-called "expected" results, obtained either by running fits on toys generated from the physics model or by using an Asimov dataset, in which statistical fluctuations are suppressed.
#
# In Combine, performing fits on toys is possible by adding the flag `-t #toys#`, while to run on an Asimov dataset we need to add `-t -1`.
# In what follows, we will perform fits on an Asimov dataset, injecting a signal strenght of $\mu=1$ via the flag `--expectSignal 1`.

# %%
from IPython.display import display, IFrame
import os

# %%
# %env PATH=opt/conda/bin/:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tmp/HiggsAnalysis/CombinedLimit/build/bin
# %env LD_LIBRARY_PATH=/opt/conda/lib/:/tmp/HiggsAnalysis/CombinedLimit/build/lib
# %env PYTHONPATH=/tmp/HiggsAnalysis/CombinedLimit/build/lib/python

# %% [markdown]
# ## Impacts

# %%
# needed because of https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/issues/1049
os.environ["CMSSW_BASE"] = "."
os.environ["SCRAM_ARCH"] = "."

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --doInitialFit -m 125 -t -1 --expectSignal 1 -n expected

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --doFits -m 125 -t -1 --expectSignal 1 -n expected

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --output impacts_expected.json -m 125 -t -1 --expectSignal 1 -n expected

# %%
# !plotImpacts.py -i impacts_expected.json -o combine_plots/impacts_expected

# %%
pdf_path = "combine_plots/impacts_expected.pdf"
display(IFrame(pdf_path, width=800, height=600))

# %% [markdown]
# ## Pre- and post-fit distributions

# %%
# !combine -M FitDiagnostics datacard_by_hand.root --saveShapes --saveWithUncertainties -n FitDiagnosticsStuffExpected -t -1 --expectSignal 1

# %%
# !for region in bin4j1b bin4j2b; do for shape in shapes_prefit shapes_fit_b shapes_fit_s; do python3 combine_scripts/postFitPlot_new.py --input_file fitDiagnosticsFitDiagnosticsStuffExpected.root --shape_type $shape --region $region --extra_suffix _expected; done; done

# %%
display(IFrame("combine_plots/stacked_plot_shapes_prefit_bin4j1b_expected.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_fit_s_bin4j1b_expected.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_prefit_bin4j2b_expected.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_fit_s_bin4j2b_expected.png", width=800, height=600))

# %% [markdown]
# ## Likelihood scan for $\mu$

# %%
# !combine -M MultiDimFit datacard_by_hand.root -n .datacard_by_hand.snapshot.expected --rMin -1 --rMax 4 --saveWorkspace -t -1 --expectSignal 1

# %%
# !combine -M MultiDimFit higgsCombine.datacard_by_hand.snapshot.expected.MultiDimFit.mH120.root -n .datacard_by_hand_expected --rMin 0 --rMax 2 --algo grid --points 80 --snapshotName MultiDimFit -t -1 --expectSignal 1

# %%
# !combine -M MultiDimFit higgsCombine.datacard_by_hand.snapshot.expected.MultiDimFit.mH120.root -n .datacard_by_hand_expected.freezeAll --rMin 0 --rMax 2 --algo grid --points 800 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -t -1 --expectSignal 1

# %%
# !python3 combine_scripts/plot1DScan.py higgsCombine.datacard_by_hand_expected.MultiDimFit.mH120.root --others 'higgsCombine.datacard_by_hand_expected.freezeAll.MultiDimFit.mH120.root:FreezeAll:2' -o combine_plots/likelihood_scan_expected --breakdown Syst,Stat

# %%
display(IFrame("combine_plots/likelihood_scan_expected.png", width=800, height=600))
