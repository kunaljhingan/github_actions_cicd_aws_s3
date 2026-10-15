from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

MODEL_PATH = Path("model_ec2_docker/score_if_else.py")


def load_scorer():
    assert MODEL_PATH.exists(), f"Missing scorer: {MODEL_PATH}"
    spec = importlib.util.spec_from_file_location("score_if_else", MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_valid_input(scorer, rows: int = 3) -> pd.DataFrame:
    feature_names = list(scorer.FEATURE_NAMES)
    row = {name: 1.0 for name in feature_names}
    return pd.DataFrame([row.copy() for _ in range(rows)])


def test_module_imports():
    assert load_scorer() is not None


def test_public_contract_exists():
    scorer = load_scorer()
    assert hasattr(scorer, "FEATURE_NAMES")
    assert isinstance(scorer.FEATURE_NAMES, (list, tuple))
    assert len(scorer.FEATURE_NAMES) > 0
    assert hasattr(scorer, "score_dataframe")
    assert callable(scorer.score_dataframe)


def test_feature_names_are_unique_non_empty_strings():
    scorer = load_scorer()
    features = list(scorer.FEATURE_NAMES)
    assert len(features) == len(set(features)), "FEATURE_NAMES contains duplicates"
    assert all(isinstance(name, str) and name.strip() for name in features)


def test_score_dataframe_smoke_contract():
    scorer = load_scorer()
    df = make_valid_input(scorer, rows=3)
    result = scorer.score_dataframe(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)

    required_outputs = {"prediction_margin", "probability", "prediction", "score"}
    assert required_outputs.issubset(result.columns)
    assert result["probability"].notna().all()
    assert result["probability"].between(0.0, 1.0).all()
    assert result["prediction"].notna().all()
    assert set(result["prediction"].astype(int).unique()).issubset({0, 1})
    assert result["prediction_margin"].notna().all()
    assert result["score"].notna().all()
    assert all(math.isfinite(float(v)) for v in result["prediction_margin"])
    assert all(math.isfinite(float(v)) for v in result["score"])


def test_scoring_is_deterministic_for_same_input():
    scorer = load_scorer()
    df = make_valid_input(scorer, rows=2)
    first = scorer.score_dataframe(df.copy())
    second = scorer.score_dataframe(df.copy())
    pd.testing.assert_frame_equal(first, second)


def test_missing_required_feature_is_rejected():
    scorer = load_scorer()
    df = make_valid_input(scorer, rows=1)
    missing_feature = scorer.FEATURE_NAMES[0]
    df = df.drop(columns=[missing_feature])

    with pytest.raises((ValueError, KeyError)) as exc:
        scorer.score_dataframe(df)

    message = str(exc.value)
    assert missing_feature in message or "missing" in message.lower()
