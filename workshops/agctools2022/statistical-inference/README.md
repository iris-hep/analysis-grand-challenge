# [Statistical inference with pyhf and cabinetry][tutorial indico]

## pyhf overview

TBD

For more information, check out the [pyhf user's guide and tutorial][pyhf tutorial].

## Distributed inference with pyhf and funcX

As pyhf is a Python **library** a user can easily write functions with its [API][pyhf API docs] that allow for specialized operations and inference.
This allows for pyhf to pair well with tools like [`funcX`][funcX docs] that allow for distributed computation to create a (fitting) Function as a Service workflow for inference.

> funcX is a distributed Function as a Service (FaaS) platform that enables flexible, scalable, and high performance remote function execution. Unlike centralized FaaS platforms, funcX allows users to execute functions on heterogeneous remote computers, from laptops to campus clusters, clouds, and supercomputers.

This workshop demo will show how to use pyhf and funcX to perform distributed statistical inference on the probability models of the [ATLAS Run 2 search for direct production of electroweakinos][1Lbb INSPIRE] [DOI: [10.1140/epjc/s10052-020-8050-3](https://doi.org/10.1140/epjc/s10052-020-8050-3)] [published on HEPData][1Lbb HEPData].

**N.B.**:
For simplicity and to focus on the end user experience, this demo will not cover deploying a `funcX` endpoint to an HPC facility (like the [University of Chicago RIVER cluster][RIVER webpage] that will be used today.)
For demonstrations that include endpoint deployment see the `funcX` resources at the end of this section.
`funcX` also requires a [Globus](https://www.globus.org/) account for data transfer.
You probably already have Globus access through your institution, so if you'd like to try this demo for yourself afterwards attempt to [login now to check](https://app.globus.org/).

### Setup

On the Open Data coffea-casa cluster being used today all the relevant dependencies are available in the default pod environment.
To **locally** create an environment with the necessary runtime dependencies create a Python virtual environment and install the dependencies defined in `requirements.txt`

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

This will:
* Locally download the pyhf pallet for the probability models from HEPData.
* Initialize the local `funcX` client.
* Serialize the inference functions.
* Dispatch the serialized function to the `funcX` endpoint to be executed as jobs as resources become available.
* Retrieve function output as the jobs finish.

Running with `time` shows that the execution wall time to send, receive, execute, and return all the results for the 125 points is under 1 minute.

The output of this run will be a JSON file `results.json` that contains the inference results for the mass hypotheses used (125 hypotheses in the case of the 1Lbb analysis).

```console
$ jq length results.json
125
```

### Visualization

To compute the exclusion contours formed from the results of the pyhf + funcX run, execute the `plot-contour.ipynb` notebook which uses the `results.json` file generated to then compute and visualize the contours. (A `results.json` has been included in the repository for example purposes for the workshop today.)

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

##### Acknowledgements

The inspiration and large parts of the source code for `exclusion` come from the following projects and people:
* The source code for the comparative visualizations in [Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods][ATL-PHYS-PUB-2019-029].
* Lukas Heinrich's [`lhoodbinder2` demonstration project](https://github.com/lukasheinrich/lhoodbinder2).
* Giordon Stark's [Reproducible ATLAS SUSY Summary Plots example GitHub Gist](https://gist.github.com/kratsg/4ff8cb2ded3b25552ff2f51cd6b854dc).
* [`HistFitter`'s `harvestToContours.py` script](https://github.com/histfitter/histfitter/blob/e85771cbd33b45d00b38326f116cffb3960f347d/scripts/harvestToContours.py), written primarily by Larry Lee.

#### Run notebook

With `exclusion` installed, run the `plot-contour.ipynb` notebook in the same directory as `results.json`.

For more information see the [funcX docs][funcx docs] and the [example code][pyhf funcx example code] for the vCHEP 2021 paper "Distributed statistical inference with pyhf enabled through funcX".

## cabinetry overview

Given time, most of the section on cabinetry will be focused on following the [cabinetry tutorials][cabinetry tutorial].
Though to tie into the `pyhf` + `funcX` demonstration, the `cabinetry-HEPData-workspace.ipynb` notebook will be using the same HEPData workspace.

For more information on cabinetry, check out the [cabinetry docs][cabinetry docs].

[tutorial indico]: https://indico.cern.ch/event/1126109/contributions/4780155/
[pyhf API docs]: https://pyhf.readthedocs.io/en/stable/api.html
[1Lbb INSPIRE]: https://inspirehep.net/literature/1755298
[1Lbb HEPData]: https://www.hepdata.net/record/ins1755298
[pyhf tutorial]: https://pyhf.github.io/pyhf-tutorial/
[funcx docs]: https://funcx.readthedocs.io/en/stable/
[ATL-PHYS-PUB-2019-029]: https://inspirehep.net/literature/1795223
[RIVER webpage]: http://river.cs.uchicago.edu/
[pyhf funcx example code]: https://github.com/matthewfeickert/distributed-inference-with-pyhf-and-funcX
[cabinetry tutorial]: https://github.com/cabinetry/cabinetry-tutorials
[cabinetry docs]: https://cabinetry.readthedocs.io/en/stable/
