"""Store round-trip tests on a synthetic match. All data here is synthetic, by design."""

import json
from pathlib import Path

import pandas as pd
import pytest
from synthetic_match import MatchSpec, make_match

from regista.store import Store, build_store
from regista.store.build import compute_tables
from regista.store.reader import StoreError

REPO = Path(__file__).resolve().parent.parent
PARAMS = json.loads((REPO / "eval" / "params_phase1.json").read_text())
CALIBRATION = json.loads((REPO / "eval" / "calibration_phase1.json").read_text())
MOMENTS = json.loads((REPO / "eval" / "params_phase2.json").read_text())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    frames = make_match(MatchSpec(period_s=90.0))
    root = tmp_path_factory.mktemp("store")
    path = build_store(frames, PARAMS, CALIBRATION, MOMENTS, "synthetic", "m1", root, window_s=60.0)
    direct = compute_tables(frames, PARAMS, CALIBRATION, MOMENTS, window_s=60.0)
    direct.pop("_radius")
    return path, direct


def test_every_table_reads_back_equal_to_direct_computation(built):
    path, direct = built
    store = Store(path, PARAMS, CALIBRATION, MOMENTS)
    assert set(store.manifest["tables"]) == set(direct)
    for name, expected in direct.items():
        got = store.table(name)
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_categorical=False, obj=name,
        )  # fmt: skip


def test_duckdb_queries_match_pandas(built):
    path, direct = built
    store = Store(path)
    counts = store.sql("SELECT team, kind, count(*) AS n FROM passes GROUP BY ALL ORDER BY ALL")
    expected = (
        direct["passes"].groupby(["team", "kind"]).size().rename("n").reset_index()
        .sort_values(["team", "kind"]).reset_index(drop=True)
    )  # fmt: skip
    pd.testing.assert_frame_equal(counts, expected, check_dtype=False)


def test_manifest_records_provenance(built):
    path, _ = built
    m = Store(path).manifest
    assert {"regista_version", "config_hash", "source", "created_at"} <= set(m)
    assert m["source"] == "synthetic" and m["match_id"] == "m1"
    assert m["tables"]["passes"] > 0


def test_store_rejects_different_parameters(built):
    path, _ = built
    changed = json.loads(json.dumps(PARAMS))
    changed["v2"]["radius_quantile"] = 0.5
    with pytest.raises(StoreError, match="different parameters"):
        Store(path, changed, CALIBRATION, MOMENTS)


def test_missing_store_raises(tmp_path):
    with pytest.raises(StoreError, match="build-store"):
        Store(tmp_path / "nope")
