# [Statistical inference with pyhf and cabinetry][tutorial indico]

## pyhf overview

TBD

For more information, check out the [pyhf user's guide and tutorial][pyhf tutorial].

## Distributed inference with pyhf and funcX

Use pyhf and funcX to perform distributed statistical inference on the probability models of the ATLAS Run 2 search for direct production of electroweakinos [DOI: [10.1140/epjc/s10052-020-8050-3](https://doi.org/10.1140/epjc/s10052-020-8050-3)] [published on HEPData][1Lbb HEPData].

### Setup

Create a Python virtual environment and install the dependencies defined in `requirements.txt`

```console
$ python -m venv venv
$ . venv/bin/activate
$ python -m pip install --upgrade pip setuptools wheel
$ python -m pip install -r requirements.txt
```

### Run

Create a file named `endpoint_id.txt` in the top level of this repository and save your funcX endpoint ID into the file.

```console
$ touch endpoint_id.txt
```

This will be read in during the run.

Pass the config JSON file for the analysis you want to run to `fit_funcx.py`

```
$ python fit_funcx.py -c config/1Lbb.json -b jax
```

```console
$ python fit_funcx.py --help
usage: fit_funcx.py [-h] [-c CONFIG_FILE] [-b BACKEND]

configuration arguments provided at run time from the CLI

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        config file
  -b BACKEND, --backend BACKEND
                        pyhf backend str alias
```

The output of this will be a JSON file `results.json` that contains the inference results for the mass hypotheses used (125 hypotheses in the case of the 1Lbb analysis).

```console
$ jq length results.json
125
```

### Visualization

To compute the exclusion contours formed from the results of the pyhf + funcX run, execute the `plot-contour.ipynb` notebook which uses the `results.json` file generated to then compute and visualize the contours.

#### Install `exclusion` library

To make the easier, a small library named `exclusion` has been created for this workshop example.
To install is, just navigate to the `exclusion/` directory and then in your Python virtual environment run

```console
$ python -m pip install .
```

and then navigate back to this top level.
If everything worked, then

```console
$ python -m pip show exclusion
```

should show `exclusion` installed in your Python virtual environment.

The inspiration and large parts of the source code come `exclusion`

For more information see the [funcX docs][funcx docs] and the [example code][pyhf funcx example code] for the vCHEP 2021 paper "Distributed statistical inference with pyhf enabled through funcX".

## cabinetry overview

Given time, most of the section on cabinetry will be focused on following the [cabinetry tutorials][cabinetry tutorial].
Though to tie into the `pyhf` + `funcX` demonstration, the `cabinetry-HEPData-workspace.ipynb` notebook will be using the same HEPData workspace.

For more information on cabinetry, check out the [cabinetry docs][cabinetry docs].

[tutorial indico]: https://indico.cern.ch/event/1126109/contributions/4780155/
[1Lbb HEPData]: https://www.hepdata.net/record/ins1755298
[pyhf tutorial]: https://pyhf.github.io/pyhf-tutorial/
[funcx docs]: https://funcx.readthedocs.io/en/stable/
[pyhf funcx example code]: https://github.com/matthewfeickert/distributed-inference-with-pyhf-and-funcX
[cabinetry tutorial]: https://github.com/cabinetry/cabinetry-tutorials
[cabinetry docs]: https://cabinetry.readthedocs.io/en/stable/
