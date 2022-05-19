"""
Orchestrate hyperparameter search for tt-bar analysis.
Uses hyperopt library to control the search over two of the analysis parameters:
    * number of histogram buckets
    * Threshold of muon and electron pt for event selection

The analysis will be run repeatedly in an attempt to optimize for ttbar_norm_bestfit, but
through a command line option can be configured to optimize for ttbar_norm_uncertainty
"""

import click
import numpy as np

from hyperopt import fmin, hp, tpe, rand

import mlflow.projects
from mlflow.tracking.client import MlflowClient

_inf = np.finfo(np.float64).max


@click.command(
    help="Perform hyperparameter search with Hyperopt library. Optimize ttbar target."
)
@click.option("--num-input-files", type=click.INT, default=10, help="input files per process, set to e.g. 10 (smaller number = faster).")
@click.option("--bin-low", type=click.INT, default=50, help="Bottom bin.")
@click.option("--bin-high", type=click.INT, default=550, help="Top bin.")
@click.option("--max-runs", type=click.INT, default=10, help="Maximum number of runs to evaluate.")
@click.option("--metric", type=click.STRING, default="ttbar_norm_bestfit", help="Metric to optimize on.")
@click.option("--algo", type=click.STRING, default="tpe.suggest", help="Optimizer algorithm.")
def train(max_runs, metric, algo, num_input_files, bin_low, bin_high):
    """
    Run hyperparameter optimization.
    """
    tracking_client = mlflow.tracking.MlflowClient()

    def new_eval(experiment_id, null_loss, return_all=False):
        """
        Create a new eval function
        :return: new eval function.
        """

        def eval(params):
            """
            Run tt-bar analysis given an ordered set of hyperparameters.

            :param params: Parameters to run the tt-bar analysis script we optimize over:
                          number of histogram bis, pt threshold for cuts
            :return: The metric value evaluated on the validation data.
            """
            import mlflow.tracking

            num_bins, pt_threshold = params
            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    uri=".",
                    entry_point="ttbar",
                    run_id=child_run.info.run_id,
                    parameters={
                        "num-input-files": str(num_input_files),
                        "num-bins": str(num_bins),
                        "bin-low": str(bin_low),
                        "bin-high": str(bin_high),
                        "pt-threshold": str(pt_threshold)
                    },
                    experiment_id=experiment_id,
                    env_manager="local",  # We are already in the environment
                    synchronous=False,  # Allow the run to fail if a model is not properly created
                )
                succeeded = p.wait()
                mlflow.log_params({"num-bins": num_bins, "pt-threshold": pt_threshold})

            if succeeded:
                analysis_run = tracking_client.get_run(p.run_id)
                metrics = analysis_run.data.metrics
                # cap the loss at the loss of the null model
                loss = min(null_loss, metrics[metric])
            else:
                # run failed => return null loss
                tracking_client.set_terminated(p.run_id, "FAILED")
                loss = null_loss

            mlflow.log_metrics(
                {
                    "metric": loss
                }
            )

            return loss

        return eval

    # Parameter search space
    space = [
        hp.choice('num-bins', np.arange(15, 50, dtype=int)),
        hp.choice("pt-threshold", np.arange(20, 30, dtype=int))
    ]

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        # Evaluate null model first.
        loss = new_eval(experiment_id, _inf, True)(params=[10, 25])
        best = fmin(
            fn=new_eval(experiment_id, loss),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))
        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
        )
        best_loss = _inf
        best_run = None
        for r in runs:
            if r.data.metrics.get(metric, _inf) < best_loss:
                best_run = r
                best_loss = r.data.metrics[metric]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                metric: best_loss,
            }
        )


if __name__ == "__main__":
    train()
