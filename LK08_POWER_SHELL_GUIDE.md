# LK-08 — GitHub-only Pipeline (Malang Solar MLOps)

This package is an **alternative to the earlier DagsHub starter**. Do not mix both workflow versions. It is intended to be extracted directly into the root of `D:\\Github\\malang-solar-mlops` on branch `feat/lk08-code-trigger`. Do not overwrite your existing LK-05/06/07 artifacts.

## 1. Source state verified

- Branch: `feat/lk08-code-trigger`
- DVC V2 CSV fixture already exists: `data/fixtures/lk08_ecmwf_history_v2.csv`
- Expected MD5: `5ed231ffe84c086adc1ee80957693395`
- Original history: 12 rows / 2 runs / 0 duplicates
- MLflow LK07 V1/V2 are local and backed up.

## 2. Add the files

Unzip this archive with **Extract Here** to the repository root. It contains `.github/workflows/`, `src/data/`, `src/model/`, `tools/`, `tests/`, and `requirements-lk08.txt`. Existing source files remain unchanged until you run the patch command. The existing fixture CSV is not included; keep the CSV already copied by the user.

## 3. Export two frozen ERA5 reference fixtures

With the local Python `.venv` activated, run from the repository root:

```powershell
python tools/export_lk08_fixtures.py
Get-ChildItem data/fixtures/lk08_era5
```

This uses your original `data/raw/era5/era5_20260928.csv` and `era5_20261009.csv`, stores only the required timestamps and GHI values, and verifies forecast MD5. The ERA5 fixture is a **historic snapshot**, not dynamic API ingestion.

If one of these files is missing, stop and restore that original reference before continuing.

## 4. Apply the small, idempotent LK06 training patch

```powershell
python tools/patch_train_lk08.py
python -m pip install -r requirements-lk08.txt
python -m pytest -q tests/test_lk08.py
```

The patch preserves default LK06 usage and adds `--no-register --output-dir reports/lk08`. It reads the MLflow tracking DB from `MLFLOW_TRACKING_URI` if set. **Do not run plain `python src/model/train.py` for LK08**, because plain invocation retains LK06 automatic registration.

## 5. Verify the full chain locally WITHOUT altering LK07 registry

Run in the same PowerShell session:

```powershell
$env:MLFLOW_TRACKING_URI = 'sqlite:///lk08-mlflow.db'
$env:MLFLOW_EXPERIMENT_NAME = 'LK08-Malang-Solar-CI'
$env:MLOPS_REGISTERED_MODEL_NAME = 'MalangSolarResidualCorrectorLK08'

python src/data/lk08_prepare.py
python src/model/train.py --no-register --output-dir reports/lk08
python src/model/lk08_evaluate.py --min-improvement 5 --max-abs-bias 20
python src/model/lk08_register.py
```

Observe the quality-gate status. A failure is a valid **guardrail**; do not falsely report success or bypass it. If passing, `STAGING MODEL LOAD: PASS` should appear. Results are in `reports/lk08/`, and the isolated DB is `lk08-mlflow.db`.

To inspect the LK08 registry locally if successful:

```powershell
python -c "import mlflow; from mlflow.tracking import MlflowClient; mlflow.set_tracking_uri('sqlite:///lk08-mlflow.db'); print(MlflowClient().get_model_version_by_alias('MalangSolarResidualCorrectorLK08','staging'))"
```

## 6. Keep generated local artifacts out of Git

Add the lines in `.gitignore-lk08-additions.txt` to the bottom of your existing `.gitignore` (do not replace the old file):

```gitignore
lk08-mlflow.db*
reports/lk08/
```

The original `mlflow.db` and LK07 registry are left untouched.

## 7. Review before committing

```powershell
python -m pytest -q tests/test_lk08.py
git status -sb
git diff -- src/model/train.py
```

Expected files to commit: workflow YAML, `data/fixtures/lk08_ecmwf_history_v2.csv`, two `data/fixtures/lk08_era5/*.csv` references, scripts under `src/`, `tools/`, tests, `requirements-lk08.txt`, `.gitignore`, and this guide. Don't add raw history, cache, `mlflow.db`, or model binaries.

```powershell
git add .github/workflows/mlops-automation.yml .gitignore requirements-lk08.txt
git add data/fixtures/lk08_ecmwf_history_v2.csv data/fixtures/lk08_era5
git add src/data/lk08_prepare.py src/model/lk08_evaluate.py src/model/lk08_register.py src/model/train.py
git add tools/export_lk08_fixtures.py tools/patch_train_lk08.py tests/test_lk08.py LK08_POWER_SHELL_GUIDE.md
git diff --cached --stat
git commit -m "feat(lk08): automate code-triggered training evaluation staging"
git push -u origin feat/lk08-code-trigger
```

## 8. GitHub Actions

Create a PR to `main`. On PR, GitHub Actions runs unit tests. Once merged into `main`, the `push` event runs the full chain and uploads evidence through `actions/upload-artifact@v4`. `workflow_dispatch` can also run the pipeline manually from `main` for diagnostics. Check `Actions` in the repository and keep a link to the complete run.

Expected evidence: (1) branch + PR; (2) MD5 matching; (3) pytest PASS; (4) automatic `push` trigger; (5) dataset bootstrap; (6) three RF MLflow runs; (7) gate checks; (8) MLflow Staging registration; (9) artifact attachment.

## Limits — describe accurately in LK08 PDF

1. The forecast and ERA5 fixture is frozen historical data. Code-triggered automated retraining is real, *dynamic ingestion and continuous updates are not* part of this CI demo.
2. Project success KPI is **>= 10% MAE improvement** on a meaningful temporal test. The separate **5% pilot gate** is provisional, and the tiny LK06 sample (6 train and 4 validation daylight rows) does not establish production quality.
3. MLflow SQLite / model artifacts exist on the ephemeral GitHub-hosted runner. Upload artifacts provide a preserved evidence snapshot, **not a persistent serving Registry** across future runs. Stage alias must be treated as a within-run demonstration.
4. The workflow is gated: failed quality gates cannot auto-register to Staging. Existing LK07 `@production` is not changed.
5. DVC history is verified against its actual `.dvc` content MD5. These fixtures are a temporary bootstrap solution, not a DVC remote.
