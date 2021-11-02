# IRIS-HEP AGC Tools 2021 Workshop, November 3rd - 4th 2021

The IRIS-HEP AGC Tools 2021 Workshop is dedicated to showcasing tools and workflows related to the so-called “Analysis Grand Challenge” (AGC) being organised by IRIS-HEP and partners. The AGC focuses on running a physics analysis at scale, including the handling of systematic uncertainties, binned statistical analysis, reinterpretation and end-to-end optimization. The AGC makes use of new and advanced analysis tools developed by the community in the Python ecosystem, and relies on the development of the required cyberinfrastructure to be executed at scale. A specific goal of the AGC is to demonstrate technologies envisioned for use at the HL-LHC.

The [agenda](https://indico.cern.ch/event/1076231/) is composed of hands-on tutorials based on various tools and services developed in the Python ecosystem by and for the particle physics community.

The following libraries and tools are included in tutorials:
* coffea-casa
* ServiceX
* func_adl
* Skyhook
* uproot
* awkward-array
* vector
* hist
* mplhep
* coffea
* cabinetry
* pyhf


Material contained in this folder (some parts are submodules, so run `git submodule update --init` to populate the folders if needed):
- `uproot-awkward-vector-demo`: data handling with uproot, awkward & vector
- `func-adl-demo`: queries with func_adl
- `ServiceX-at-IRIS-HEP-ACG-workshop-2021`: data delivery with ServiceX
- `HZZ_analysis_pipeline`: analysis pipeline demonstration featuring ServiceX, coffea, cabinetry & pyhf
