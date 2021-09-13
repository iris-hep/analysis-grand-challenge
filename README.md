# Analysis Grand Challenge (AGC)

IRIS-HEP is organizing an “Analysis Grand Challenge”, which includes the binned analysis, reinterpretation and end-to-end optimization physics analysis use cases and developing needed cyber infrastructure to execute them, in order to demonstrate technologies envisioned for HL-LHC.  To enable these use cases and more, the expected capabilities include (examples are new services under development by various groups):

* New user interfaces: Complementary services that present the analyst with a notebook-based interface.  Example software: Jupyter.
* Data access: Services that provide quick access to the experiment’s official data sets, often allowing simple derivations and local caching for efficient access.  Example software and services: Rucio, ServiceX, SkyHook, iDDS, RNTuple.
* Event selection: Systems/frameworks allowing analysts to process entire datasets, select desired events, and calculate derived quantities.  Example software and services: Coffea, awkward-array, func_adl, RDataFrame.
Histogramming and summary statistics: Closely tied to the event selection, histogramming tools provide physicists with the ability to summarize the observed quantities in a dataset.  Example software and services: Coffea, func_adl, cabinetry, hist.
* Statistical model building and fitting: Tools that translate specifications for event selection, summary statistics, and histogramming quantities into statistical models, leveraging the capabilities above, and perform fits and statistical analysis with the resulting models.  Example software and services: cabinetry, pyhf, FuncX+pyhf fitting service
* Reinterpretation / analysis preservation:  Standards for capturing the entire analysis workflow, and services to reuse the workflow which enables reinterpretation.  Example software and services: REANA, RECAST.

Analysis Grand Challenge will be conducted during 2021‒2023, leaving enough time for tuning software tools and services developed as a part of the IRIS-HEP ecosystem before the start-up of the HL-LHC.

To learn more about AGC please check [dedicated webpage](https://iris-hep.org/grand-challenges.html).

