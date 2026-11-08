"""Idempotently patch the repo's existing LK06 train.py for safe LK08 CI.

Run from repository root: python tools/patch_train_lk08.py
Original LK06 default behavior is preserved; --no-register defers registration
until independent LK08 evaluation succeeds.
"""
from __future__ import annotations

from pathlib import Path

PATH = Path('src/model/train.py')


def replace_once(text: str, before: str, after: str) -> str:
    n = text.count(before)
    if n != 1:
        raise RuntimeError(f'Expected exactly one match (got {n}) for: {before[:110]!r}')
    return text.replace(before, after, 1)


def main() -> None:
    source = PATH.read_text(encoding='utf-8-sig')
    if "parser.add_argument('--no-register'" in source:
        print('LK08 patch already applied; no changes.')
        return
    updated = replace_once(source, 'import json\nimport re\n', 'import argparse\nimport json\nimport os\nimport re\n')
    updated = replace_once(updated, "TRACKING_URI = 'sqlite:///mlflow.db'",
        "TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')")
    updated = replace_once(updated, "EXPERIMENT_NAME = 'LK06-Malang-Solar-Residual-Correction'",
        "EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'LK06-Malang-Solar-Residual-Correction')")
    updated = replace_once(updated, "REGISTERED_MODEL_NAME = 'MalangSolarResidualCorrector'",
        "REGISTERED_MODEL_NAME = os.getenv('MLOPS_REGISTERED_MODEL_NAME', 'MalangSolarResidualCorrector')")
    updated = replace_once(updated, 'def main() -> None:\n    train, valid = load_and_split()',
        "def main() -> None:\n    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--no-register', action='store_true', help='CI: register only after independent evaluation gate')\n"
        "    parser.add_argument('--output-dir', type=Path, default=None)\n"
        "    args = parser.parse_args()\n    output_dir = args.output_dir or OUTPUT_DIR\n    train, valid = load_and_split()")
    updated = replace_once(updated, '    mlflow.set_tracking_uri(TRACKING_URI)\n    mlflow.set_experiment(EXPERIMENT_NAME)',
        '    mlflow.set_tracking_uri(TRACKING_URI)\n    mlflow.set_registry_uri(TRACKING_URI)\n    mlflow.set_experiment(EXPERIMENT_NAME)')
    before = (
        "    registered = mlflow.register_model(\n"
        "        model_uri=best['model_uri'],\n"
        "        name=REGISTERED_MODEL_NAME,\n"
        "        await_registration_for=30,\n"
        "    )\n"
        "    MlflowClient().set_registered_model_alias(\n"
        "        REGISTERED_MODEL_NAME, 'lk06-candidate', registered.version\n"
        "    )\n"
    )
    after = (
        "    registered_version = None\n"
        "    registered_alias = None\n"
        "    if not args.no_register:\n"
        "        registered = mlflow.register_model(\n"
        "            model_uri=best['model_uri'],\n"
        "            name=REGISTERED_MODEL_NAME,\n"
        "            await_registration_for=30,\n"
        "        )\n"
        "        MlflowClient().set_registered_model_alias(\n"
        "            REGISTERED_MODEL_NAME, 'lk06-candidate', registered.version\n"
        "        )\n"
        "        registered_version = registered.version\n"
        "        registered_alias = 'lk06-candidate'\n"
    )
    updated = replace_once(updated, before, after)
    updated = replace_once(updated,
        "                'scope': 'LK06 demonstration (not production benchmark)',",
        "                'scope': ('LK08 CI demo, frozen DVC v2 dataset, not production benchmark' "
        "if args.no_register else 'LK06 demonstration (not production benchmark)'),")
    updated = replace_once(updated, "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
        "    output_dir.mkdir(parents=True, exist_ok=True)")
    updated = replace_once(updated,
        "    ranking.to_csv(OUTPUT_DIR / 'lk06_experiment_summary.csv', index=False)",
        "    summary_file = output_dir / ('experiment_summary.csv' if args.no_register else 'lk06_experiment_summary.csv')\n"
        "    ranking.to_csv(summary_file, index=False)")
    updated = replace_once(updated, "'registered_version': registered.version", "'registered_version': registered_version")
    updated = replace_once(updated, "'registered_alias': 'lk06-candidate'", "'registered_alias': registered_alias")
    updated = replace_once(updated,
        "    (OUTPUT_DIR / 'lk06_best_model.json').write_text(",
        "    metadata_file = output_dir / ('best_model.json' if args.no_register else 'lk06_best_model.json')\n"
        "    metadata_file.write_text(")
    updated = replace_once(updated,
        "    print('Metadata saved: reports/lk06_best_model.json')",
        "    print(f'Metadata saved: {metadata_file}')")
    updated = replace_once(updated,
        "    print(f'\\nRegistered candidate: {REGISTERED_MODEL_NAME} v{registered.version}')",
        "    if args.no_register:\n"
        "        print('CI training complete: registration deferred to LK08 evaluation gate.')\n"
        "    else:\n"
        "        print(f'\\nRegistered candidate: {REGISTERED_MODEL_NAME} v{registered_version}')")
    import ast
    ast.parse(updated, filename=str(PATH))
    PATH.write_text(updated, encoding='utf-8')
    print('Patched src/model/train.py successfully; default LK06 behavior preserved.')


if __name__ == '__main__':
    main()
