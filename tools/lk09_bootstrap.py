"""Initialize a clean shared MLflow registry once; preserve it across restarts."""
from __future__ import annotations

import os
import subprocess
import sys

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
NAME = os.environ['MLOPS_REGISTERED_MODEL_NAME']
MODEL_URI = f'models:/{NAME}@staging'


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        existing = client.get_model_version_by_alias(NAME, 'staging')
    except MlflowException as exc:
        if exc.error_code != 'RESOURCE_DOES_NOT_EXIST':
            raise
    else:
        mlflow.pyfunc.load_model(MODEL_URI)
        print(f'LK09 BOOTSTRAP: REUSED existing {MODEL_URI}, version={existing.version}', flush=True)
        return

    stages = [
        [sys.executable, 'src/data/lk08_prepare.py'],
        [sys.executable, 'src/model/train.py', '--no-register', '--output-dir', 'reports/lk08'],
        [sys.executable, 'src/model/lk08_evaluate.py', '--min-improvement', '5', '--max-abs-bias', '20'],
        [sys.executable, 'src/model/lk08_register.py'],
    ]
    for command in stages:
        print('LK09 INIT:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)
    candidate = client.get_model_version_by_alias(NAME, 'staging')
    mlflow.pyfunc.load_model(MODEL_URI)
    print(f'LK09 BOOTSTRAP: PASS model={NAME} version={candidate.version} alias=staging', flush=True)


if __name__ == '__main__':
    main()
