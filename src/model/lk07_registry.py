"""LK-07: MLflow model versioning, staging, production alias, and inference smoke test.

Copy to src/model/lk07_registry.py. Run from the repository root on the same
machine/environment that holds the LK-06 mlflow.db and model artifacts.

Commands:
  python src/model/lk07_registry.py status
  python src/model/lk07_registry.py register-v2
  python src/model/lk07_registry.py stage --version 1
  python src/model/lk07_registry.py production --version 1
  python src/model/lk07_registry.py infer

Legacy stages are deprecated in MLflow. We attempt the legacy transition for
coursework evidence; the 'staging'/'production' aliases are the inference source
of truth. Promotion here is a *local lifecycle simulation*, not deployment.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# train.py is in the same src/model/ folder.
from train import FEATURE_COLUMNS, load_and_split, provenance

ROOT = Path(__file__).resolve().parents[2]
NAME = "MalangSolarResidualCorrector"
TRACKING_URI = "sqlite:///mlflow.db"
V1_REPORT = ROOT / "reports/lk06_best_model.json"
MANIFEST = ROOT / "reports/lk07_model_registry.yaml"
INFERENCE_RESULT = ROOT / "reports/lk07_inference_result.json"
# The LK-07 challenger is deliberately different from LK-06 V1.
V2_PARAMS = {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 1, "random_state": 42}


def setup() -> MlflowClient:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"Run commands from the project root: {ROOT}")
    if not (ROOT / "mlflow.db").is_file():
        raise SystemExit("Local mlflow.db not found. This must run where LK-06 was executed.")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    c = MlflowClient()
    try:
        c.get_registered_model(NAME)
        c.get_model_version(NAME, "1")
    except Exception as exc:
        raise SystemExit(f"LK-06 registry model Version 1 not found: {exc}") from exc
    return c


def get_versions(c: MlflowClient) -> list:
    return sorted(c.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version))


def alias_version(c: MlflowClient, alias: str) -> int | None:
    try:
        return int(c.get_model_version_by_alias(NAME, alias).version)
    except MlflowException:
        return None


def current_dvc_hash() -> str:
    return provenance()["dvc_forecast_md5"]


def check_lk06_provenance() -> None:
    if not V1_REPORT.is_file():
        raise SystemExit("Missing reports/lk06_best_model.json (created by LK-06).")
    meta = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    if meta.get("registered_model") != NAME or int(meta.get("registered_version", -1)) != 1:
        raise SystemExit("LK-06 model metadata does not identify expected model/version.")
    stored = meta.get("dvc_forecast_md5")
    actual = current_dvc_hash()
    if stored != actual:
        raise SystemExit(
            f"DVC version changed since LK-06: LK06={stored}; current={actual}. "
            "Restore/check dataset before comparing model versions."
        )


def version_metrics(c: MlflowClient, v) -> dict:
    try:
        run = c.get_run(v.run_id)
    except Exception:
        return {"run_id": str(v.run_id), "val_mae": None, "params": {}, "dvc_forecast_md5": ""}
    return {
        "run_id": run.info.run_id,
        "val_mae": run.data.metrics.get("val_mae"),
        "val_rmse": run.data.metrics.get("val_rmse"),
        "params": run.data.params,
        "dvc_forecast_md5": run.data.tags.get("dvc_forecast_md5", ""),
        "git_commit": run.data.tags.get("git_commit", ""),
    }


def status(c: MlflowClient) -> None:
    print(f"=== LK-07 MODEL REGISTRY: {NAME} ===")
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"DVC forecast MD5: {current_dvc_hash()}")
    for v in get_versions(c):
        m = version_metrics(c, v)
        print(
            f"v{v.version} | stage={getattr(v, 'current_stage', 'unknown')} "
            f"| aliases={list(getattr(v, 'aliases', []))} "
            f"| val_mae={m['val_mae']} | run_id={m['run_id']}"
        )
    print(f"Staging alias : v{alias_version(c, 'staging')}")
    print(f"Production alias: v{alias_version(c, 'production')}")


def write_manifest(c: MlflowClient) -> None:
    """Write lightweight YAML with the DVC content hash and registry provenance.

    Uses only stdlib; JSON strings are valid YAML 1.2 scalar syntax.
    This is Git-tracked metadata, NOT an MLflow or DVC model binary.
    """
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    versions = get_versions(c)
    chosen = alias_version(c, "production")
    def q(x: object) -> str:
        return json.dumps(str(x) if x is not None else "", ensure_ascii=False)
    lines = [
        "# LK-07: Git-track this YAML; actual artifacts are stored in local MLflow.",
        f"updated_at_utc: {q(datetime.now(timezone.utc).isoformat())}",
        f"model_name: {q(NAME)}",
        f"tracking_uri: {q(TRACKING_URI)}",
        "dataset:",
        f"  dvc_pointer: {q('data/raw/ecmwf_ifs_history.csv.dvc')}",
        f"  dvc_md5: {q(current_dvc_hash())}",
        f"  labeled_dataset: {q('data/processed/training/ecmwf_ifs_labeled.parquet')}",
        "lifecycle:",
        f"  staging_version: {alias_version(c, 'staging') or 'null'}",
        f"  production_version: {chosen if chosen else 'null'}",
        f"  production_uri: {q(f'models:/{NAME}@production') if chosen else 'null'}",
        "  local_simulation_only: true",
        "versions:",
    ]
    for v in versions:
        m = version_metrics(c, v)
        lines += [
            f"  - version: {int(v.version)}",
            f"    run_id: {q(m['run_id'])}",
            f"    stage: {q(getattr(v, 'current_stage', 'unknown'))}",
            f"    aliases: [{', '.join(q(a) for a in getattr(v, 'aliases', []))}]",
            f"    validation_mae: {m['val_mae'] if m['val_mae'] is not None else 'null'}",
            f"    validation_rmse: {m.get('val_rmse') if m.get('val_rmse') is not None else 'null'}",
            f"    source_dvc_md5: {q(m.get('dvc_forecast_md5') or current_dvc_hash())}",
            f"    training_git_commit: {q(m.get('git_commit', ''))}",
            f"    model_uri: {q(f'models:/{NAME}/{v.version}')}",
            "    parameters:",
        ]
        for key, value in sorted(m["params"].items()):
            lines.append(f"      {key}: {q(value)}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Manifest saved: {MANIFEST.relative_to(ROOT)}")


def register_v2(c: MlflowClient) -> None:
    check_lk06_provenance()
    existing = get_versions(c)
    if any(int(v.version) >= 2 for v in existing):
        raise SystemExit(
            "Version 2 or newer already exists. Registration intentionally stopped "
            "to avoid silently creating V3. Run 'status' and inspect MLflow UI."
        )
    train, valid = load_and_split()
    X_train = train[FEATURE_COLUMNS].astype(float)
    y_train = train["residual"].astype(float)
    X_valid = valid[FEATURE_COLUMNS].astype(float)
    actual = valid["actual_ghi"].to_numpy(dtype=float)
    raw = valid["raw_ghi_forecast"].to_numpy(dtype=float)
    baseline = float(mean_absolute_error(actual, raw))
    print(f"Training {len(train)} rows; validating {len(valid)} rows.")
    print("WARNING: this is an LK-07 lifecycle simulation with a very small dataset.")
    mlflow.set_experiment("LK07-Malang-Solar-Model-Lifecycle")
    model = RandomForestRegressor(**V2_PARAMS, n_jobs=-1)
    with mlflow.start_run(run_name="LK07-RF-150-depth5-v2") as run:
        mlflow.log_params(V2_PARAMS)
        mlflow.set_tags({
            **provenance(), "task": "ghi_residual_regression",
            "dataset": "LK05 DVC forecast history + ERA5",
            "model_version_intent": "lk07_v2_challenger",
            "evaluation": "temporal_validation_by_forecast_run",
            "scope": "local lifecycle simulation, not production validation",
        })
        model.fit(X_train, y_train)
        pred_residual = model.predict(X_valid)
        corrected = np.clip(raw + pred_residual, 0, None)
        mae = float(mean_absolute_error(actual, corrected))
        rmse = float(math.sqrt(mean_squared_error(actual, corrected)))
        bias = float(np.mean(corrected - actual))
        improvement = float(100 * (baseline - mae) / baseline) if baseline else 0.0
        mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse,
                            "val_bias": bias, "baseline_raw_mae": baseline,
                            "improvement_percent": improvement})
        info = mlflow.sklearn.log_model(
            sk_model=model, name="residual_model", input_example=X_train.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"V2 candidate run: {run.info.run_id}")
        print(f"V2 validation: MAE={mae:.3f}; RMSE={rmse:.3f}; bias={bias:+.3f} W/m²")
        registered = mlflow.register_model(info.model_uri, NAME, await_registration_for=30)
    if int(registered.version) != 2:
        raise RuntimeError(f"Expected version 2, got {registered.version}; inspect registry.")
    c.set_model_version_tag(NAME, "2", "dvc_forecast_md5", current_dvc_hash())
    c.set_model_version_tag(NAME, "2", "lifecycle_role", "challenger")
    c.set_registered_model_alias(NAME, "lk07-challenger", "2")
    print("Registered Version 2 with alias @lk07-challenger.")
    status(c)
    write_manifest(c)


def transition(c: MlflowClient, version: int, target: str) -> None:
    check_lk06_provenance()
    v = c.get_model_version(NAME, str(version))
    if target == "Production":
        staged = alias_version(c, "staging")
        if staged != version:
            raise SystemExit(
                f"Cannot promote v{version}: @staging currently points to v{staged}. "
                f"First run: python src/model/lk07_registry.py stage --version {version}"
            )
    m = version_metrics(c, v)
    print(f"Selected v{version}: validation MAE={m['val_mae']} W/m²")
    print("This is a LOCAL COURSEWORK lifecycle simulation, not real deployment.")
    # Legacy stages were deprecated; keep evidence where the local MLflow supports them.
    try:
        updated = c.transition_model_version_stage(
            name=NAME, version=str(version), stage=target,
            archive_existing_versions=(target == "Production"),
        )
        print(f"Legacy MLflow stage transition: {updated.current_stage}")
    except (MlflowException, AttributeError, NotImplementedError) as exc:
        print(f"Legacy stage unavailable: {exc}")
        print("Using MLflow aliases instead (supported lifecycle mechanism).")
    alias = "staging" if target == "Staging" else "production"
    c.set_registered_model_alias(NAME, alias, str(version))
    if target == "Production":
        c.set_registered_model_alias(NAME, "champion", str(version))
    c.set_model_version_tag(NAME, str(version), "lifecycle_status", target.lower())
    c.set_model_version_tag(NAME, str(version), "lifecycle_scope", "LK07_local_simulation")
    print(f"Alias @{alias} now points to v{version}.")
    if target == "Production":
        print(f"Inference URI: models:/{NAME}@production")
    status(c)
    write_manifest(c)


def infer(c: MlflowClient) -> None:
    production_version = alias_version(c, "production")
    if production_version is None:
        raise SystemExit("No production alias found. Complete Staging -> Production first.")
    _, valid = load_and_split()
    X_sample = valid[FEATURE_COLUMNS].astype(float).head(1)
    if X_sample.empty:
        raise SystemExit("No validation sample available for inference smoke test.")
    uri = f"models:/{NAME}@production"
    print(f"Loading {uri} (resolved v{production_version})")
    pyfunc = mlflow.pyfunc.load_model(uri)
    arr = np.asarray(pyfunc.predict(X_sample), dtype=float).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        raise ValueError(f"Invalid model output: {arr!r}")
    row = valid.iloc[0]
    raw = float(row["raw_ghi_forecast"])
    residual = float(arr[0])
    corrected = max(0.0, raw + residual)
    result = {
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_model": NAME, "production_alias_version": production_version,
        "model_uri": uri, "forecast_valid_time_utc": str(row["valid_time_utc"]),
        "raw_forecast_ghi_w_m2": raw,
        "predicted_residual_w_m2": residual,
        "corrected_forecast_ghi_w_m2": corrected,
        "era5_reference_w_m2_for_offline_check_only": float(row["actual_ghi"]),
        "dvc_forecast_md5": current_dvc_hash(),
        "status": "PASS", "note": "Single local smoke test, not service deployment",
    }
    INFERENCE_RESULT.parent.mkdir(parents=True, exist_ok=True)
    INFERENCE_RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("INFERENCE SMOKE TEST: PASS")
    print(f"Evidence saved: {INFERENCE_RESULT.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "register-v2", "infer"):
        sub.add_parser(name)
    for name in ("stage", "production"):
        sp = sub.add_parser(name)
        sp.add_argument("--version", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()
    c = setup()
    if args.command == "status":
        status(c)
    elif args.command == "register-v2":
        register_v2(c)
    elif args.command == "stage":
        transition(c, args.version, "Staging")
    elif args.command == "production":
        transition(c, args.version, "Production")
    elif args.command == "infer":
        infer(c)


if __name__ == "__main__":
    main()
