CMS Open Data $t\\bar{t}$: from data delivery to statistical inference

We are using [2015 CMS Open Data](https://cms.cern/news/first-cms-open-data-lhc-run-2-released) 
in this demonstration to showcase an analysis pipeline. It features data 
delivery and processing, histogram construction and visualization, as well as 
statistical inference.

This notebook was developed in the context of the 
[IRIS-HEP AGC tools 2022 workshop](https://indico.cern.ch/e/agc-tools-2). This 
work was supported by the U.S. National Science Foundation (NSF) Cooperative 
Agreement OAC-1836650 (IRIS-HEP).

This is a technical demonstration. We are including the relevant workflow 
aspects that physicists need in their work, but we are not focusing on making 
every piece of the demonstration physically meaningful. This concerns in 
particular systematic uncertainties: we capture the workflow, but the actual 
implementations are more complex in practice. If you are interested in the 
physics side of analyzing top pair production, check out the latest results from 
ATLAS and CMS! If you would like to see more technical demonstrations, also 
check out an ATLAS Open Data example demonstrated previously.

## Tracking Analysis Runs with MLFlow
A version of this analysis has been instrumented with 
[MLFlow](https://mlflow.org) to record runs of this analysis along with the 
input parameters, the fit results, and generated plots. To use the tracking
service you will need:
* Conda
* Access to an MLFlow tracking service instance
* Environment variables set to allow the script to communicate with the tracking service and the back-end object store:
  * `MLFLOW_TRACKING_URI`
  * `MLFLOW_S3_ENDPOINT_URL`
  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`

If you would like to install a local instance of the MLFlow tracking service on
you Kubernetes cluster, this 
[helm chart](https://artifacthub.io/packages/helm/ncsa/mlflow) is a good start.

For reproducibility, MLFlow insists on running the analysis in a conda 
environment. This is defined in `conda.yaml`. 

The MLFlow project is defined in `MLprojec` - this file specifies two different
_entrypoints_ 

`ttbar` is the entrypoint for running a single analysis. It offers a number
of command line parameters to control the analysis. It can be run as
```shell
mlflow run -P num-bins=25 -P pt-threshold=25 .
```
### Hyperparameter Searches
MLFlow is often used in optimizing models by running with different 
hyperparamters until a minimal loss function is realized. We've borrowed this
approach for optimizing an analysis. You can orchestrate a number of analysis 
runs with different input settings by using the `hyperopt` entrypoint.

```shell
 mlflow run -e hyperopt -P max_runs=20 .
```
