"""LK-08: register ONLY a candidate that passed the independent quality gate.

Production alias is intentionally untouched. Legacy Staging stage is best effort;
MLflow's @staging alias is the durable CI inference contract.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

GATE = Path('reports/lk08/evaluation.json')
TRAINING = Path('reports/lk08/best_model.json')
OUTPUT = Path('reports/lk08/registration.json')


def main() -> None:
    gate = json.loads(GATE.read_text(encoding='utf-8'))
    if gate.get('passed') is not True or not all(gate.get('checks', {}).values()):
        raise SystemExit('Quality gate not passed: registration blocked.')
    meta = json.loads(TRAINING.read_text(encoding='utf-8'))
    if gate.get('candidate_run_id') != meta.get('run_id'):
        raise SystemExit('Run ID mismatch: registration blocked.')
    if gate.get('dvc_forecast_md5') != meta.get('dvc_forecast_md5'):
        raise SystemExit('DVC version mismatch: registration blocked.')

    uri = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    name = os.environ.get('MLOPS_REGISTERED_MODEL_NAME', 'MalangSolarResidualCorrectorLK08')
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    c = MlflowClient()

    v = mlflow.register_model(model_uri=meta['model_uri'], name=name, await_registration_for=60)
    c.set_model_version_tag(name, v.version, 'validation_status', 'passed')
    c.set_model_version_tag(name, v.version, 'lifecycle_status', 'Staging')
    c.set_model_version_tag(name, v.version, 'source_dvc_md5', meta['dvc_forecast_md5'])
    c.set_model_version_tag(name, v.version, 'scope', 'LK08 demo, not production approval')
    c.set_model_version_tag(name, v.version, 'git_commit', str(meta['git_commit']))

    legacy_stage = 'not_set_deprecated_or_unsupported'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            c.transition_model_version_stage(name, str(v.version), 'Staging')
        legacy_stage = 'Staging'
    except Exception as exc:
        print(f'Legacy stage unavailable ({type(exc).__name__}); using @staging alias and lifecycle_status tag.', flush=True)

    c.set_registered_model_alias(name, 'staging', str(v.version))
    resolved = c.get_model_version_by_alias(name, 'staging')
    if str(resolved.version) != str(v.version):
        raise RuntimeError('Staging alias does not resolve to newly registered version.')

    # Verify the registered artifact is actually loadable on this runner.
    staging_uri = f'models:/{name}@staging'
    mlflow.pyfunc.load_model(staging_uri)
    print('STAGING MODEL LOAD: PASS', flush=True)

    report = {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'registered_model': name,
        'registered_version': str(v.version),
        'legacy_stage': legacy_stage,
        'alias': 'staging',
        'model_uri': f'models:/{name}@staging',
        'source_run_id': meta['run_id'],
        'source_dvc_md5': meta['dvc_forecast_md5'],
        'validation_status': 'passed',
        'scope': 'Staging candidate only; existing @production not changed',
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('=== LK08 AUTO REGISTRY ===', flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print('LK08 REGISTRY: STAGING PASS', flush=True)


if __name__ == '__main__':
    main()
