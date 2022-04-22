# [Statistical inference with pyhf and cabinetry][tutorial indico]

## pyhf overview

TBD

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
$ python fit_funcx.py -c config/1Lbb.json -b numpy
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

For more information see https://github.com/matthewfeickert/distributed-inference-with-pyhf-and-funcX

## cabinetry overview

[tutorial indico]: https://indico.cern.ch/event/1126109/contributions/4780155/
[1Lbb HEPData]: https://www.hepdata.net/record/ins1755298
