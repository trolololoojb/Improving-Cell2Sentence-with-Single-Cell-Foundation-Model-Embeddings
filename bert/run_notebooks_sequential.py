#!/usr/bin/env python3
"""Run two notebooks sequentially: second starts only after first succeeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "mammography_bert_distil.ipynb",
    ROOT / "mammography_bert_distil_classWeight_newLoss.ipynb",
]


def run_notebook(notebook: Path) -> None:
    if not notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=python3",
        "--ExecutePreprocessor.timeout=-1",
        str(notebook),
    ]

    print(f"\n[RUN] {notebook.name}")
    subprocess.run(cmd, check=True)
    print(f"[OK ] {notebook.name}")


def main() -> int:
    try:
        for notebook in NOTEBOOKS:
            run_notebook(notebook)
    except subprocess.CalledProcessError as exc:
        print(
            f"\n[FAIL] Notebook execution failed with exit code {exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1

    print("\n[DONE] Both notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
