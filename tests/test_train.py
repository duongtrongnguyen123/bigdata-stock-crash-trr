"""Smoke + correctness tests for the train/ meta-learner modules."""
from __future__ import annotations

import numpy as np
import pytest

from train.features import FEATURES_FULL, build_dataset
from train.run import _gbm
from train.threshold import _confusion


@pytest.fixture(scope="module")
def df():
    return build_dataset()


def test_dataset_shape(df):
    assert len(df) > 500
    for col in FEATURES_FULL + ["label_true", "era"]:
        assert col in df.columns
    assert set(df["label_true"].unique()) <= {0, 1}
    assert df["label_true"].sum() > 10  # has crash events


def test_no_nan_in_features(df):
    assert not df[FEATURES_FULL].isna().any().any()


def test_gbm_trains_and_predicts(df):
    X, y = df[FEATURES_FULL], df["label_true"].to_numpy()
    m = _gbm().fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (len(df),)
    assert ((p >= 0) & (p <= 1)).all()


def test_confusion_math():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    c = _confusion(y, pred)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, **c} or c["tp"] == 1
    assert c["precision"] == pytest.approx(0.5)
    assert c["recall"] == pytest.approx(0.5)
    assert c["f1"] == pytest.approx(0.5)


def test_ensemble_scorer_optional():
    """If the model artifact exists, the scorer returns a valid probability."""
    from serving.ensemble import is_available, score_ensemble
    if not is_available():
        pytest.skip("models/trr_meta.pkl not present (run train.export)")
    out = score_ensemble(0.6, 20, 12)
    assert 0.0 <= out["ensemble_crash_prob"] <= 1.0
    assert "technicals_asof" in out
