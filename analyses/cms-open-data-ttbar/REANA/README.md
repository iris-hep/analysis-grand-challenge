# REANA example - AGC CMS ttbar analysis with Coffea

This demo shows the submission of [AGC](https://arxiv.org/abs/1010.2506) - Analysis Grand Challenge
to the [REANA](http://www.reana.io/) using the Snakemake as an workflow engine.

## Analysis Grand Challenge

For full explanation please have a look at this documentation:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7274936.svg)](https://doi.org/10.5281/zenodo.7274936)
[![Documentation Status](https://readthedocs.org/projects/agc/badge/?version=latest)](https://agc.readthedocs.io/en/latest/?badge=latest)

The Analysis Grand Challenge (AGC) is about performing the last steps in an analysis pipeline at scale to test workflows envisioned for the HL-LHC.
This includes

- columnar data extraction from large datasets,
- processing of that data (event filtering, construction of observables, evaluation of systematic uncertainties) into histograms,
- statistical model construction and statistical inference,
- relevant visualizations for these steps,

The physics analysis task is a $t\bar{t}$ cross-section measurement with 2015 CMS Open Data (see `datasets/cms-open-data-2015`).
The current reference implementation can be found in `analyses/cms-open-data-ttbar`.

### 1. Input data 

We are using [2015 CMS Open Data](https://cms.cern/news/first-cms-open-data-lhc-run-2-released) in this demonstration to showcase an analysis pipeline. The input `.root` files are located in the  `nanoAODschema.json`.
## Analysis Code
The current coffea AGC version defines the coffea Processor, which includes a lot of the physics analysis details:
- event filtering and the calculation of observables,
- event weighting,
- calculating systematic uncertainties at the event and object level,
- filling all the information into histograms that get aggregated and ultimately returned to us by coffea.

The analysis takes the following inputs:

- ``nanoAODschema.json`` input `.root` files
- ``Snakefile`` The Snakefile for 
- ``ttbar_analysis_reana.ipynb`` The main notebook file where files are processed and analysed.
- ``file_merging.ipynb`` Notebook to merge each processed `.root` file in one file with unique keys.
- ``final_merging.ipynb`` Notebook to merge histograms together all of 

### 2. Compute environment 

To be able to rerun the AGC after some time, we need to
"encapsulate the current compute environment", for example to freeze the ROOT version our
analysis is using. We shall achieve this by preparing a [Docker](https://www.docker.com/)
container image for our analysis steps.

We are using the modified verison of the ``analysis-systems-base`` [Docker image](https://github.com/iris-hep/analysis-systems-base) container with additional packages, the main on is [papermill](https://papermill.readthedocs.io/en/latest/) which allows to run the Jupyter Notebook from the command line with additional parameters.

In our case, the Dockerfile creates a conda virtual environment with all necessary packages for running the AGC analysis.

```console
$ less environment/Dockerfile
```
Let's go inside the environment and build it
```console
$ cd environment/
```

We can build our AGC environment image and give it a name
`docker.io/reanahub/reana-demo-agc-cms-ttbar-coffea`:

```console
$ docker build -t docker.io/reanahub/reana-demo-agc-cms-ttbar-coffea .
```

We can push the image to the DockerHub image registry:

```console
$ docker push docker.io/reanahub/reana-demo-agc-cms-ttbar-coffea
```

### 3. Kerberos authentication
Some data are located at the eos/public so in order to process the big amount of files, user should be authenticated with Kerberos.
In our case we achieve it by setting up:
```console
workflow:
  type: snakemake
  resources:
    kerberos: true
  file: Snakefile
```
If you are pocessing small amount of files (less than 10) you can set this option to `False`.
Or you can also set the kerberos authentication via the Snakemake rules.
For deeper understanding please refer to the (REANA documentation)[https://docs.reana.io/advanced-usage/access-control/kerberos/]

### 4. AGC workflow for Snakemake suitability  
REANA provides support for the Snakemake workflow engine. To ensure the fastest execution of the AGC ttbar workflow, a two-level (multicascading) parallelization approach with Snakemake is implemented.
In the initial step, Snakemake distributes all jobs across separate nodes, each with a single `.root` file for `ttbar_analysis_reana.ipynb`. 
Subsequently, after the completion of each rule, the merging of individual files into one per sample takes place.
#Here is the high level of AGC workflow 

```console
                                +-----------------------------------------+
                                | Take the CMS open data from nanoaod.json|
                                +-----------------------------------------+
                                                    |
                                                    |
                                                    |
                                                    v  
                                  +-----------------------------------+
                                  |rule: Process each file in parallel|
                                  +-----------------------------------+
                                                    |
                                                    |
                                                    |
                                                    v
                                +-----------------------------------------+                
                                |rule: Merge created files for each sample|
                                +-----------------------------------------+  
                                                    |
                                                    |
                                                    |
                                                    v
                                +----------------------------------------------+ 
                                |rule: Merge sample files into single histogram| 
                                +----------------------------------------------+
```

### 5. Running the AGC on REANA

The [reana.yaml](reana.yaml) file describes the above analysis
structure with its inputs, code, runtime environment, computational workflow steps and
expected outputs:

```yaml
version: 0.8.0
inputs:
  files:
    - ttbar_analysis_reana.ipynb 
    - nanoaod_inputs.json
    - fix-env.sh
    - corrections.json
    - Snakefile
    - file_merging.ipynb
    - final_merging.ipynb
    - prepare_workspace.py

  directories:
    - histograms
    - utils
workflow:
  type: snakemake
  resources:
    kerberos: true
  file: Snakefile
outputs:
  files:
    - histograms_merged.root
```

We can now install the REANA command-line client, run the analysis and download the
resulting plots:

```console
$ # create new virtual environment
$ virtualenv ~/.virtualenvs/reana
$ source ~/.virtualenvs/reana/bin/activate
$ # install REANA client
$ pip install reana-client
$ # connect to some REANA cloud instance
$ export REANA_SERVER_URL=https://reana.cern.ch/
$ export REANA_ACCESS_TOKEN=XXXXXXX
$ # run AGC workflow
$ reana-client run -w reana-agc-cms-ttbar-coffea
$ # ... should be finished in around 6 minutes if you select all files in the Snakefile
$ reana-client status
$ # list workspace files
$ reana-client ls
```

Please see the [REANA-Client](https://reana-client.readthedocs.io/) documentation for
more detailed explanation of typical `reana-client` usage scenarios.

### 6. Output results

The output is created under the name of ``histograms_merged.root`` which can be further evaluated with variety of AGC tools.




