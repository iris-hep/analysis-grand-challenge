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
#     version: 3.12.5
# ---

# %% [markdown]
# # Statistical Inference with Combine

# %% [markdown]
# [Combine](https://arxiv.org/abs/2404.06614) is the tool used in CMS to perform statistical inference and produce the results that we publish in our papers.
# Based on RooStats and RooFit, it is adopted by more than 90% of the analyses performed in the past few years.
# Despite a steep learning curve, it provides everything needed to publish a CMS analysis.
# In this notebook we will use the histograms produced in the previous step to perform statistical inference and make statements on the parameters of the theory we are testing.

# %% [markdown]
# As you can see from the [documentation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/), the tool mostly relies on a list of scripts with multiple options that have to be called from the terminal.

# %%
from IPython.display import display, IFrame
import os

# %%
os.makedirs("combine_plots", exist_ok=True)

# %%
# %env PATH=opt/conda/bin/:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tmp/HiggsAnalysis/CombinedLimit/build/bin
# %env LD_LIBRARY_PATH=/opt/conda/lib/:/tmp/HiggsAnalysis/CombinedLimit/build/lib
# %env PYTHONPATH=/tmp/HiggsAnalysis/CombinedLimit/build/lib/python

# %% [markdown]
# ## Write the datacard
#
# Combine uses a specific way to describe the likelihood, the so-called datacard. It can be written by hand (which is convenient only when we deal with rather simple analyses) or automate the process based on the coffea output.
# At the moment, each analysis seem to have its own way to write the datacard based on the coffea outputs. An attempt to realize a "once for all" procedure is being made in PocketCoffea.

# %%
# !cat datacard_by_hand.txt

# %% [markdown]
# The first part
# ```
# imax *
# jmax *
# kmax *
#
# ```
# refers to the number of channels, backgrounds and nuisances parameters respectively. It is possible to add `*` and let the tool figure out on its own, but it fails if this part is not there...

# %% [markdown]
# The lines in this part
# ```
# shapes * * all_histograms_fps4_$CHANNEL.root $PROCESS $PROCESS_$SYSTEMATIC
# shapes ttbar * all_histograms_fps4_$CHANNEL.root ttbar ttbar_$SYSTEMATIC
# ```
# follow the following format
# ```
# shapes process channel file histogram [histogram_with_systematics]
# ```
# and point to the histograms that we produced in the previous part.

# %% [markdown]
# The part
# ```
# bin          bin4j1b      bin4j2b
# observation  -1           -1
# ```
# defines the observed rate per channel. One can either write it explicitly or make Combine figure out on its own by integrating (like in this case, by adding `-1`).

# %% [markdown]
# The part
# ```
# bin      bin4j1b  bin4j1b              bin4j1b              bin4j1b          bin4j1b  bin4j2b  bin4j2b              bin4j2b              bin4j2b          bin4j2b
# process  ttbar    single_top_s_chan    single_top_t_chan    single_top_tW    wjets    ttbar    single_top_s_chan    single_top_t_chan    single_top_tW    wjets
# process  0        1                    2                    3                4        0        1                    2                    3                4
# rate     -1       -1                   -1                   -1               -1       -1       -1                   -1                   -1               -1
# ```
# defines what is signal and what is backgound, essentially by assigning numbers `<-0` to signal processes and `>0` to backgrounds, and the expected rate (also in this case, a value of `-1` tells Combine to figure out by integrating.

# %% [markdown]
# The last section refers to systematic uncertainties. 
#
# The block
# ```
# pt_scale   shape 1 1 1 1 1 1 1 1 1 1
# pt_res     shape 1 1 1 1 1 1 1 1 1 1
# btag_var_0 shape 1 1 1 1 1 1 1 1 1 1
# btag_var_1 shape 1 1 1 1 1 1 1 1 1 1
# btag_var_2 shape 1 1 1 1 1 1 1 1 1 1
# btag_var_3 shape 1 1 1 1 1 1 1 1 1 1
# ME_var     shape 1 0 0 0 0 1 0 0 0 0
# PS_var     shape 1 0 0 0 0 1 0 0 0 0
# scale      shape 1 0 0 0 0 1 0 0 0 0
# scale_var  shape 0 0 0 0 1 0 0 0 0 1
# ```
# is for shape uncertainties (notice that they correspond to the $\pm 1 \sigma$ histograms that we generated during the coffea part.
#
# The block
# ```
# lumi       lnN   1.03 1.03 1.03 1.03 1.03 1.03 1.03 1.03 1.03 1.03
# ```
# describes the type of distribution (log-normal, `lnN`) used to described the nuisance parameter associated to luminosity uncertainty. The following `#channels*#processes` define the relative effect on the rate of each process in each channel (in this case a 3% uncertainty).
#
# The last block
# ```
# bin4j1b autoMCStats 0
# bin4j2b autoMCStats 0
# ```
# refers to the statistical uncertainties coming from the amount of events in MC samples. It is described at [this link](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/bin-wise-stats/).

# %% [markdown]
# ## Create workspace
#
# Combine uses a specific script called `text2workspace.py` to create the workspace used for the fit. 
# The part
# ```
# --PO 'map=.*/ttbar:r[1.0,0.0,3.0]'
# ```
# scales the signal contribution with a signal strength modifier `r` (usually called $\mu$ in the literature).

# %%
# !text2workspace.py datacard_by_hand.txt --PO 'map=.*/ttbar:r[1.0,0.0,3.0]'

# %% [markdown]
# ## Impacts and pulls
#
# In most (all) the analyses approval procedures, it is required to check the impact of the nuisance parameters (NP) on the parameter of interest ($\mu$).
# The impact of a NP is defined as the shift $ \Delta \mu$ induced as the NP is fixed to its $\pm 1 \sigma$ values, with all the other parameters profiled as normal.
#
# Another quantity worth checking is the pull, defined as $pull(\theta) = \frac{\hat{\theta} - \theta_0}{\sigma_0}$, which quantifies how far from its expected value we had to "pull" $\theta$ while finding the MLE.

# %%
# needed because of https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/issues/1049
os.environ["CMSSW_BASE"] = "."
os.environ["SCRAM_ARCH"] = "."

# %% [markdown]
# The procedure follows these steps:
# 1. do an initial fit of the POIs with the `--robustFit 1` option

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --doInitialFit -m 125

# %% [markdown]
# 2. perform a scan for each NP

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --doFits -m 125

# %% [markdown]
# 3. collect and save the output to a JSON file

# %%
# !combineTool.py -M Impacts -d datacard_by_hand.root --robustFit 1 --output impacts.json -m 125

# %% [markdown]
# 4. make a plot

# %%
# !plotImpacts.py -i impacts.json -o combine_plots/impacts

# %%
pdf_path = "combine_plots/impacts.pdf"
display(IFrame(pdf_path, width=800, height=600))

# %% [markdown]
# ## Pre- and post-fit distributions
#
# One of the modes Combine can be run with is `FitDiagnostics`. It performs a background-only and a signal+background fit and produces information useful for disgnostics.
# One thing that one can do consists in visualizing the distributions we are fitting both before and after the fit (pre- and post- fit).

# %%
# !combine -M FitDiagnostics datacard_by_hand.root --saveShapes --saveWithUncertainties -n FitDiagnosticsStuff

# %% [markdown]
# After running the fits, we can plot the distributions with one of the scripts taken from the Combine tutorial and lcoated inside `combine_scripts`.

# %%
# !for region in bin4j1b bin4j2b; do for shape in shapes_prefit shapes_fit_b shapes_fit_s; do python3 combine_scripts/postFitPlot_new.py --input_file fitDiagnosticsFitDiagnosticsStuff.root --shape_type $shape --region $region; done; done

# %%
display(IFrame("combine_plots/stacked_plot_shapes_prefit_bin4j1b.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_fit_s_bin4j1b.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_prefit_bin4j2b.png", width=800, height=600))
display(IFrame("combine_plots/stacked_plot_shapes_fit_s_bin4j2b.png", width=800, height=600))

# %% [markdown]
# ## Likelihood scan for $\mu$
#
# Another step commonly performed consists in performing an explicit scan of the profile likelihood to provide an interval at the $\alpha$CL around the best fit.
# In asymptotic approximation, this done by identifying the parameter values at which the test statistic $q = -2\Delta lnL$ equals a critical value. This value is the $\alpha$ quantile of the $\chi^2$ distribution with one degree of freedom.
#
# The first part consists in performing a global fit whose result we save in a workspace.

# %%
# !combine -M MultiDimFit datacard_by_hand.root -n .datacard_by_hand.snapshot --rMin -1 --rMax 4 --saveWorkspace

# %% [markdown]
# Using the previous result as a starting point, we then perform a scan in a range of the POI.

# %%
# !combine -M MultiDimFit higgsCombine.datacard_by_hand.snapshot.MultiDimFit.mH120.root -n .datacard_by_hand --rMin 0 --rMax 2 --algo grid --points 80 --snapshotName MultiDimFit

# %% [markdown]
# In order to discriminate the uncertainty contribution between systematic and statistical, we perform another scan but this time we "freeze" the nuisance parameters to their best fit value.

# %%
# !combine -M MultiDimFit higgsCombine.datacard_by_hand.snapshot.MultiDimFit.mH120.root -n .datacard_by_hand.freezeAll --rMin 0 --rMax 2 --algo grid --points 800 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances

# %% [markdown]
# We can then use the scripts provided in the Combine tutorial to get the final results.

# %%
# !python3 combine_scripts/plot1DScan.py higgsCombine.datacard_by_hand.MultiDimFit.mH120.root --others 'higgsCombine.datacard_by_hand.freezeAll.MultiDimFit.mH120.root:FreezeAll:2' -o combine_plots/likelihood_scan --breakdown Syst,Stat

# %%
display(IFrame("combine_plots/likelihood_scan.png", width=800, height=600))
