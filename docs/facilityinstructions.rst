Running at Analysis Facilities 
===============================================================

Here is some information about running the notebook at different analysis facilities (more will be added soon).

Coffea-Casa at UNL
---------------------------------------------------------------
To use this facility, follow the instructions at this site: `https://coffea-casa.readthedocs.io/en/latest/cc_user.html <https://coffea-casa.readthedocs.io/en/latest/cc_user.html>`_. This facility includes `ServiceX`, `Triton`, and will include `mlflow`. In the file `config.yaml`, ensure that the following parameters are set:

* `AF: coffea_casa`
* TRITON_URL: `agc-triton-inference-server:8001`


Elastic Analysis Facility at Fermilab
---------------------------------------------------------------
To use this facility, follow the instructions at this site: `https://eafjupyter.readthedocs.io/en/latest/ <https://eafjupyter.readthedocs.io/en/latest/>`_. A Fermilab account is required to access this facility. This facility includes `Triton`. In the file `config.yaml`, ensure that the following parameters are set:

* `AF: EAF`
* `TRITON_URL: triton.apps.okddev.fnal.gov:443`
* `USE_SERVICEX: False`

`mlflow` Instructions
---------------------------------------------------------------

If you want to try using `mlflow` in the training pipeline on an analysis facility that does not include `mlflow`, you can use a local setup as described in this documentation: `https://mlflow.org/docs/latest/quickstart.html <https://mlflow.org/docs/latest/quickstart.html>`_. NSCA also has a good set-up if you do not want to store models locally, and you can sign up for an account here: `https://wiki.ncsa.illinois.edu/display/NCSASoftware/MLFlow\+at\+NCSA <https://wiki.ncsa.illinois.edu/display/NCSASoftware/MLFlow\+at\+NCSA>`_.