from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(".")
MODEL_PATH = REPO_ROOT / "model_ec2_docker" / "score_if_else.py"


def test_model_file_exists():
    assert MODEL_PATH.is_file(), f"Expected model file at {MODEL_PATH}"


def test_model_file_compiles():
    subprocess.run([sys.executable, "-m", "py_compile", str(MODEL_PATH)], check=True)


def test_workflow_yaml_is_parseable():
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    assert workflow_dir.is_dir(), "Missing .github/workflows directory"
    workflow_files = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
    assert workflow_files, "No GitHub Actions workflow files found"

    for workflow in workflow_files:
        with workflow.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
        assert isinstance(parsed, dict), f"{workflow} is not valid workflow YAML"


def test_no_obvious_aws_access_key_is_committed():
    access_key_pattern = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
    ignored_dirs = {".git", ".venv", "venv", "__pycache__"}

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_dirs for part in path.parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not access_key_pattern.search(text), f"Possible AWS access key committed in {path}"


@pytest.mark.parametrize("candidate", [Path("run_scoring.sh"), Path("scripts/run_scoring.sh")])
def test_shell_script_syntax_if_present(candidate: Path):
    if not candidate.exists():
        pytest.skip(f"{candidate} is not present in this repository")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available on this machine")
    subprocess.run([bash, "-n", str(candidate)], check=True)
