.. image:: cms-open-data-ttbar/utils/ecosystem.png
   :width: 95%
   :alt: AGC Pipeline
   :align: center

Analysis Grand Challenge Documentation
===============================================================

|Binder| |DOI|

Introduction
---------------------------------------------------------------

The Analysis Grand Challenge (AGC) is about performing the last steps in an analysis pipeline at scale to test workflows envisioned for the HL-LHC.
This includes

* columnar data extraction from large datasets,
* processing of that data (event filtering, construction of observables, evaluation of systematic uncertainties) into histograms,
* statistical model construction and statistical inference,
* relevant visualizations for these steps,

all done in a reproducible & preservable way that can scale to HL-LHC requirements.

The AGC has two major pieces:

#. **specification of a physics analysis** using Open Data which captures relevant workflow aspects encountered in physics analyses performed at the LHC,
#. a **reference implementation** demonstrating the successful execution of this physics analysis at scale.

The physics analysis task is a :math:`t\bar{t}` cross-section measurement with 2015 CMS Open Data (see `datasets/cms-open-data-2015 <https://github.com/iris-hep/analysis-grand-challenge/tree/main/datasets/cms-open-data-2015>`_).
The current reference implementation can be found in `analyses/cms-open-data-ttbar <https://github.com/iris-hep/analysis-grand-challenge/tree/main/analyses/cms-open-data-ttbar>`_.
In addition to this, `analyses/atlas-open-data-hzz <https://github.com/iris-hep/analysis-grand-challenge/tree/main/analyses/atlas-open-data-hzz>`_ contains a smaller scale :math:`H\rightarrow ZZ^*` analysis based on ATLAS Open Data.

More information & references
---------------------------------------------------------------

The `AGC website <https://agc.readthedocs.io/>`_ contains more information about the AGC.

The project has been described in a few conference proceedings.
If you make use of the AGC in your research, please consider citing the project.
We recommend citing the 2022 ICHEP proceedings when referring to the project more generally and the other other publications as relevant to the specific context.

- **ICHEP 2022 proceedings** with a general introduction: `DOI: 10.22323/1.414.0235 <https://doi.org/10.22323/1.414.0235>`_, `INSPIRE <https://inspirehep.net/literature/2598292>`_
- **ACAT 2022 proceedings** with first performance measurements: IOP Conference Series publication forthcoming, `arXiv:2304.05214 [hep-ex] <https://arxiv.org/abs/2304.05214>`_, `INSPIRE <https://inspirehep.net/literature/2650460>`_
- **CHEP 2023 procedings** with an overview: `DOI: 10.1051/epjconf/202429506016 <https://doi.org/10.1051/epjconf/202429506016>`_, `INSPIRE <https://inspirehep.net/literature/2743608>`_
- **CHEP 2023 procedings** focused on ML task: `DOI: 10.1051/epjconf/202429508011 <https://doi.org/10.1051/epjconf/202429508011>`_, `INSPIRE <https://inspirehep.net/literature/2743013>`_

Additional information is available in a series of AGC-focused workshops:

- `IRIS-HEP AGC Tools 2021 Workshop, Nov 3-4 2021 <https://indico.cern.ch/e/agc-tools-workshop>`_
- `IRIS-HEP AGC Tools 2022 Workshop, April 25-26 2022 <https://indico.cern.ch/e/agc-tools-2>`_
- `IRIS-HEP AGC workshop 2023, May 3-5 2023 <https://indico.cern.ch/e/agc-workshop-2023>`_
- `IRIS-HEP AGC Demonstration 2023, Sep 14 2023 <https://indico.cern.ch/e/agc-demonstration>`_

We also have a `dedicated IRIS-HEP webpage <https://iris-hep.org/grand-challenges.html>`_.

AGC and IRIS-HEP
---------------------------------------------------------------

The AGC serves as an integration exercise for IRIS-HEP, allowing the testing of new services, libraries and workflows on dedicated analysis facilities in the context of realistic physics analyses.

AGC and you
---------------------------------------------------------------

We believe that the AGC can be useful in various contexts:

* testbed for software library development,
* realistic environment to prototype analysis workflows,
* functionality, integration & performance test for analysis facilities.

We are very interested in seeing (parts of) the AGC implemented in different ways.
Please get in touch if you have investigated other approaches you would like to share!
There is no need to implement the full analysis task — it splits into pieces (for example the production of histograms) that can also be tackled individually.

AGC implementations and related projects
---------------------------------------------------------------

Besides the implementation in this repository, have a look at the following implementations as well:

* ROOT RDataFrame-based implementation: `root-project/analysis-grand-challenge <https://github.com/root-project/analysis-grand-challenge>`_
* pure Julia implementation: `Moelf/LHC_AGC.jl <https://github.com/Moelf/LHC_AGC.jl>`_
* columnflow implementation: `columnflow/agc_cms_ttbar <https://github.com/columnflow/agc_cms_ttbar>`_

Additional related projects are listed below.
Are we missing some things in this list?
Please get in touch!

- AGC on REANA with Snakemake: `iris-hep/agc-reana <https://github.com/iris-hep/agc-reana>`_
- small demo of AGC with `dask-awkward` and `coffea` 2024: `iris-hep/calver-coffea-agc-demo <https://github.com/iris-hep/calver-coffea-agc-demo/>`_
- columnar analysis with ATLAS PHYSLITE Open Data: `iris-hep/agc-physlite <https://github.com/iris-hep/agc-physlite>`_
- exploring automatic differentiation for physics analysis: `iris-hep/agc-autodiff <https://github.com/iris-hep/agc-autodiff>`_
- AGC data processing with RNTuple files: `iris-hep/agc-rntuple <https://github.com/iris-hep/agc-rntuple>`_

More details: what is being investigated in the AGC context
---------------------------------------------------------------

* **New user interfaces**: Complementary services that present the analyst with a notebook-based interface.  Example software: Jupyter.
* **Data access**: Services that provide quick access to the experiment’s official data sets, often allowing simple derivations and local caching for efficient access.  Example software and services: ``Rucio``, ``ServiceX``, ``SkyHook``, ``iDDS``, ``RNTuple``.
* **Event selection**: Systems/frameworks allowing analysts to process entire datasets, select desired events, and calculate derived quantities.  Example software and services: ``coffea``, ``awkward-array``, ``func_adl``, ``RDataFrame``.
* **Histogramming and summary statistics**: Closely tied to the event selection, histogramming tools provide physicists with the ability to summarize the observed quantities in a dataset.  Example software and services: ``coffea``, ``func_adl``, ``cabinetry``, ``hist``.
* **Statistical model building and fitting**: Tools that translate specifications for event selection, summary statistics, and histogramming quantities into statistical models, leveraging the capabilities above, and perform fits and statistical analysis with the resulting models.  Example software and services: ``cabinetry``, ``pyhf``, ``FuncX`` + ``pyhf`` fitting service
* **Reinterpretation / analysis preservation**:  Standards for capturing the entire analysis workflow, and services to reuse the workflow which enables reinterpretation.  Example software and services: ``REANA``, ``RECAST``.

Acknowledgements
---------------------------------------------------------------

This work was supported by the U.S. National Science Foundation (NSF) cooperative agreements `OAC-1836650 <https://nsf.gov/awardsearch/showAward?AWD_ID=1836650>`_ and `PHY-2323298 <https://nsf.gov/awardsearch/showAward?AWD_ID=2323298>`_ (IRIS-HEP).


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: ttbar with CMS Open Data
   :glob:

   cms-open-data-ttbar/ttbar_analysis_pipeline
   versionsdescription
   facilityinstructions
   taskbackground
   event-plotting/plotEvents

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: H>ZZ* with ATLAS Open Data
   :glob:

   atlas-open-data-hzz/HZZ_analysis_pipeline



Search
===============================================================

* :ref:`search`

.. |DOI| image:: https://zenodo.org/badge/239605861.svg
   :target: https://zenodo.org/record/7274937#.ZBmGj-zML9E
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/alexander-held/PyHEP-2022-AGC/main?labpath=talk.ipynb
