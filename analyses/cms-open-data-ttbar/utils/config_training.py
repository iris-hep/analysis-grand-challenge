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
        "MLFLOW_TRACKING_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ6eXdjRVdfd2hOSzBTMDJLS3Nxd0Q0cGNjTXJpc1BSUUJDcEo4T0o1Rm40In0.eyJleHAiOjE2ODg3NTY3ODUsImlhdCI6MTY4ODcyMDc4OCwiYXV0aF90aW1lIjoxNjg4NzIwNzg1LCJqdGkiOiI2MzRmNWYzNy1mMTQ2LTQ0OTEtYmQyMC01MmNmM2QyZWE4NzAiLCJpc3MiOiJodHRwczovL2tleWNsb2FrLnNvZnR3YXJlLWRldi5uY3NhLmlsbGlub2lzLmVkdS9yZWFsbXMvbWxmbG93IiwiYXVkIjpbIm1sZmxvdy1kZW1vIiwiYWNjb3VudCJdLCJzdWIiOiIyOTdhM2ExNS02Nzc0LTQ0NDItOGVjNy0zNzBlNmVhNzM2MWIiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJtbGZsb3ctZGVtbyIsInNlc3Npb25fc3RhdGUiOiIyMTdjZTc3Yi1iZDc0LTQ2ZjUtOWZmNi04Mzc5OThlYjY3NmQiLCJzY29wZSI6InByb2ZpbGUgZ3JvdXBzIGVtYWlsIiwic2lkIjoiMjE3Y2U3N2ItYmQ3NC00NmY1LTlmZjYtODM3OTk4ZWI2NzZkIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJFbGxpb3R0IEthdWZmbWFuIiwiZ3JvdXBzIjpbIi9ncnBfamlyYV91c2VycyIsIi9zZF9tbGZsb3ciLCIvYWxsX3VzZXJzIiwiL2ppcmEtdXNlcnMiLCIvYWxsX2hwY191c2VyX3NwbyIsIm9mZmxpbmVfYWNjZXNzIiwiZGVmYXVsdC1yb2xlcy1tbGZsb3ciLCJ1bWFfYXV0aG9yaXphdGlvbiJdLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJla2F1ZmZtYSIsImdpdmVuX25hbWUiOiJFbGxpb3R0IiwiZmFtaWx5X25hbWUiOiJLYXVmZm1hbiIsImVtYWlsIjoiZWxtYWthODcwMEBnbWFpbC5jb20ifQ.AL87yHriW1Sfj7vBju1PTJDHf4qm-IX429GhDJ7ZZEpO9zDENE2oAJmvqeOgXX9bVlJ14NcxaCR08L0O3rKoBk9NVVJT25bR7I1jAQvO2HQxCSnkYFGzKlLN3wZHzoJ6UN_DoMmdPDxMFcn_5n5HZDLccApB-5BOpB2Gp_PYYOuOdGKXBvN1uIrsa9tl526XX5t4aiGW-lRyvTcmpwF7up22Xq6Va-J3_O85RdTkm45NSl7EMTCc_hjjpmAj6icMD2q_PDr0jBsfWyyTxLq7x4vj6cvBUF8y4XjcA9uiqcJhC8g0jIAzUK0zpJniNa6AKNHYpiAIAe1jweLkKsWDgA",
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