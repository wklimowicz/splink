"""Seeded equality against v5.0.0 for the issue-3367 experiment."""
import importlib.util
import subprocess
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pytest

from splink import DuckDBAPI, Linker, SettingsCreator
import splink.comparison_library as cl
from splink.internals import estimate_u as candidate


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    path = tmp_path_factory.mktemp("u_reference") / "estimate_u.py"
    path.write_bytes(subprocess.check_output([
        "git", "show", "v5.0.0:splink/internals/estimate_u.py"
    ]))
    spec = importlib.util.spec_from_file_location("splink.internals.u_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["HEAD", "u-performance-arrow", "u-performance-aggregation"])
def optimized(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("u_variant") / "estimate_u.py"
    path.write_bytes(subprocess.check_output([
        "git", "show", f"{request.param}:splink/internals/estimate_u.py"
    ]))
    spec = importlib.util.spec_from_file_location("splink.internals.u_variant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_linker(link_type, n_inputs):
    api = DuckDBAPI(duckdb.connect(config={"threads": 2}))
    frames = []
    for source in range(n_inputs):
        data = [
            {"unique_id": i, "name": None if i % 11 == 0 else ["ANNA", "ANNE", "EMMA", "JOHN"][i % 4],
             "nino": None if i % 13 == 0 else f"AB{i % 41:06d}{'B' if (i+source)%7 == 0 else 'A'}"}
            for i in range(160 + source * 17)
        ]
        frames.append(api.register(pa.Table.from_pylist(data), dataset_display_name=f"source_{source}"))
    settings = SettingsCreator(
        link_type=link_type,
        comparisons=[cl.LevenshteinAtThresholds("name", [1, 2]), cl.LevenshteinAtThresholds("nino", [1, 2]),
            {"output_column_name": "fixed_name", "comparison_levels": [
                {"sql_condition": "name_l IS NULL OR name_r IS NULL", "is_null_level": True},
                {"sql_condition": "name_l = name_r", "u_probability": 0.125, "fix_u_probability": True},
                {"sql_condition": "ELSE", "u_probability": 0.875, "fix_u_probability": True},
            ]},
        ],
        blocking_rules_to_generate_predictions=[],
    )
    linker = Linker(frames, settings, log_level="ERROR")
    return linker, api


def run(module, link_type, n_inputs, chunks, minimum):
    linker, api = make_linker(link_type, n_inputs)
    counts = []
    original = module._MUCountsAccumulator.update_from_chunk_counts
    def capture(self, rows):
        counts.append((self._output_column_name, sorted(
            (int(r["comparison_vector_value"]), float(r["u_count"])) for r in rows
        )))
        return original(self, rows)
    try:
        with patch.object(module._MUCountsAccumulator, "update_from_chunk_counts", capture):
            module.estimate_u_values(linker, max_pairs=6000, seed=271828,
                                     num_chunks=chunks, min_count_per_level=minimum)
        linker._populate_m_u_from_trained_values()
        model = linker._settings_obj.as_dict()["comparisons"]
        assert not any(name.startswith("__splink__df_concat_sample") for (name,) in api.duckdb_con.sql("SHOW TABLES").fetchall())
        return counts, model
    finally:
        api.duckdb_con.close()


@pytest.mark.parametrize("link_type,n_inputs", [("dedupe_only",1), ("link_only",2), ("link_only",3), ("link_and_dedupe",2)])
@pytest.mark.parametrize("chunks", [1, 10])
@pytest.mark.parametrize("minimum", [None, 5])
def test_sample_counts_and_models_match_v500(reference, optimized, link_type, n_inputs, chunks, minimum):
    assert run(optimized, link_type, n_inputs, chunks, minimum) == run(reference, link_type, n_inputs, chunks, minimum)


def test_arrow_sample_is_removed_after_failure():
    linker, api = make_linker("link_only", 2)
    try:
        with patch.object(candidate, "_accumulate_u_counts_from_chunk_and_check_min_count", side_effect=RuntimeError("test failure")):
            with pytest.raises(RuntimeError, match="test failure"):
                candidate.estimate_u_values(linker, max_pairs=6000, seed=271828)
        names = [name for (name,) in api.duckdb_con.sql("SHOW TABLES").fetchall()]
        assert not any(name.startswith("__splink__df_concat_sample") for name in names)
        assert not any(name.startswith("__splink__df_concat_sample") for name in names)
    finally:
        api.duckdb_con.close()
