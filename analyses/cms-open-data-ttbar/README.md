# CMS Open Data $t\bar{t}$

This directory is focused on running the CMS Open Data $t\bar{t}$ analysis through the `ttbar_analysis_pipeline.ipynb` notebook or `ttbar_analysis_pipeline.py` Python script. Here is a brief description of files in this repository:

| File Name                     | Description                                                                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| GetIOBranches.py              | This script uses `nanoaod_branch_ratios.json` to calculate a list of branches to read in specific fractions of data for I/O benchmarking tests. |
| cabinetry_config.yml          | This is a config file for `cabinetry`, which is used to build a statistical workspace.                                                          |
| cabinetry_config_ml.yml       | This is another `cabinetry` config file, but specifically for the Machine Learning observables.                                                 |
| coffea.ipynb                  | This was the old name of the `ttbar_analysis_pipeline.ipynb` notebook (moved to avoid confusion with `coffea` library)                          |
| corrections.json              | `correctionlib` file to handle correction factors in the $t\bar{t}$ analysis.                                                                   |
| make_corrections_json.py      | Python script used to write `corrections.json`                                                                                                  |
| nanoaod_branch_ratios.json    | Branch size fractions for all files listed in `nanoaod_inputs.json` combined.                                                                   |
| nanoaod_inputs.json           | `xrootd` links to all files used in the $t\bar{t}$ analysis.                                                                                    |
| requirements.txt              | Requirements to run analysis.                                                                                                                   |
| ttbar_analysis_pipeline.ipynb | Notebook version of the analysis.                                                                                                               |
| ttbar_analysis_pipeline.py    | Python script version of the analysis (linked to notebook via `jupytext`)                                                                       |
| models/                       | Contains models used for ML inference task (when `USE_TRITON = False`)                                                                          |
| utils/                        | Contains code for bookkeeping and cosmetics, as well as some boilerplate. Also contains images used in notebooks.                               |
| utils/config.py               | This is a general config file to handle different options for running the analysis.                               |

#### Instructions for paired notebook

If you only care about running the `ttbar_analysis_pipeline.ipynb` notebook, you can completely ignore the `ttbar_analysis_pipeline.py` file.

This notebook (`ttbar_analysis_pipeline.ipynb`) is paired to the file `ttbar_analysis_pipeline.py` via Jupytext (https://jupytext.readthedocs.io/en/latest/). Using `git diff` with this file instead of the `.ipynb` file is much simpler, as you don't have to deal with notebook metadata or output images. However, in order for the notebook output to be preserved, the notebook still needs to be version controlled. It is ideal to run `git diff` with the option `-- . ':(exclude)*.ipynb'`, so that `.ipynb` files are ignored. 

The `.py` file can also be run as a Python script.

There is a pre-commit hook that will ensure that the files are synced via Jupytext. If you are cloning the repository for the first time, you must run
```
pre-commit install
```
in the GitHub repository. After this, you should be able to commit with the Jupytext pre-commit hook! You may have to commit twice, as the pre-commit hook may make changes to the notebook file.

The `ttbar_analysis_pipeline.ipynb` and `ttbar_analysis_pipeline.py` files should auto-sync with each other, but in case they don't, you can force a sync by running the command
```
jupytext --sync ttbar_analysis_pipeline.ipynb
```

If you wish to create a new notebook (`notebook.ipynb`) and pair it with a .py file via Jupytext, you can either click View &rArr; Activate Command Palettte, then click "Pair Notebook with percent Script", or run the command
```
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

#### Validating outputs

The `validate_histograms.py` script can be used to check that the histograms produced by the analysis are consistent with expectations.
To use it, simply call `python validate_histograms.py --reference reference/histos_1_file_per_process.json`, where the last
argument is the appropriate reference file for the number of files per process and the kind of run that has been performed.
For full usage help see the output of `python validate_histograms.py --help`.

`validate_histograms.py` can also be used to create new references by passing the `--dump-json` option.
