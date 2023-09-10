from .config import config as config_main

config = {
    "global": {
        # analysis facility: set to "coffea_casa" for coffea-casa environments, "EAF" for FNAL, "local" for local setups
        "AF": "coffea_casa",
        
    },
    "benchmarking": {
        # chunk size to use
        "CHUNKSIZE": 100000,
        # scaling for local setups with FuturesExecutor
        "NUM_CORES": 4,
    },
    "ml": {
        # fill with token needed to publish results to mlflow server
        "MLFLOW_TRACKING_TOKEN": "",
        # fill with mlflow tracking URI (ex: https://mlflow-demo.software-dev.ncsa.illinois.edu)
        "MLFLOW_TRACKING_URI": "https://mlflow-demo.software-dev.ncsa.illinois.edu",
        # name of mlflow experiment (for tracking, not for model publishing)
        "MLFLOW_EXPERIMENT_NAME": "optimize-reconstruction-bdt-00",
        # number of folds for cross-validation
        "N_FOLD": 2,
        # number of trials (per model) for hyperparameter optimization. Total number of trials will be 2*N_TRIALS
        "N_TRIALS": 5,
        # name to use for registering model in mlflow
        "MODEL_NAME": "reconstruction_bdt_xgb"
    },
}

# get some ML options from main config (alter in one place only!)
config["ml"]["FEATURE_NAMES"] = config_main["ml"]["FEATURE_NAMES"]
config["ml"]["FEATURE_DESCRIPTIONS"] = config_main["ml"]["FEATURE_DESCRIPTIONS"]
config["ml"]["BIN_LOW"] = config_main["ml"]["BIN_LOW"]