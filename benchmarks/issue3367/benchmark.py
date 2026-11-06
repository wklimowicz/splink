"""Paired u-estimation benchmark; caller supplies an untrained real-data Linker.

Variant files are estimate_u.py from pinned control/Arrow/aggregation/combined refs.
Only aggregate counts, probabilities and process measurements leave this function.
"""
import hashlib
import importlib.util
import json
import logging
import statistics
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import duckdb
import psutil
import pyarrow


class PeakRSS:
    def __enter__(self):
        self.process = psutil.Process()
        self.start = self.process.memory_info().rss
        self.peak = self.start
        self.stopped = threading.Event()
        def poll():
            while not self.stopped.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stopped.set()
        self.thread.join()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_u_benchmark(linker, variant_dir, *, max_pairs, min_count_per_level,
                    num_chunks=10, repetitions=2, seed=271828):
    variants = {}
    sources = {}
    for label in ["control", "arrow", "aggregation", "combined"]:
        path = Path(variant_dir) / f"{label}.py"
        name = f"splink.internals._u_benchmark_{label}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        variants[label] = module
        sources[label] = hashlib.sha256(path.read_bytes()).hexdigest()

    results = []
    counts_by_variant = {}
    u_by_variant = {}
    for repeat in range(repetitions):
        order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
        for label in order:
            module = variants[label]
            trial = deepcopy(linker)
            counts = []
            original = module._MUCountsAccumulator
            class CaptureCounts(original):
                def update_from_chunk_counts(self, rows):
                    counts.append((self._output_column_name, sorted(
                        (int(r["comparison_vector_value"]), int(r["u_count"]))
                        for r in rows
                    )))
                    return super().update_from_chunk_counts(rows)
            module._MUCountsAccumulator = CaptureCounts
            logger = logging.getLogger(module.__name__)
            old_level = logger.level
            logger.setLevel(logging.ERROR)
            try:
                with PeakRSS() as memory:
                    wall, cpu = time.perf_counter(), time.process_time()
                    module.estimate_u_values(
                        trial, max_pairs=max_pairs, seed=seed,
                        min_count_per_level=min_count_per_level,
                        num_chunks=num_chunks,
                    )
                    cpu, wall = time.process_time()-cpu, time.perf_counter()-wall
                trial._populate_m_u_from_trained_values()
                probabilities = [
                    {"comparison": c.output_column_name,
                     "levels": [(lev.comparison_vector_value, lev._u_probability,
                                 lev._trained_u_probabilities)
                                for lev in c._comparison_levels_excluding_null]}
                    for c in trial._settings_obj.comparisons
                ]
                row = {
                    "variant": label, "repeat": repeat,
                    "wall_seconds": round(wall, 4), "cpu_seconds": round(cpu, 4),
                    "cpu_over_wall": round(cpu/wall, 2),
                    "rss_start_gib": round(memory.start/2**30, 3),
                    "rss_peak_gib": round(memory.peak/2**30, 3),
                    "chunk_count": len(counts), "counts_sha256": digest(counts),
                    "u_sha256": digest(probabilities),
                }
                counts_by_variant.setdefault(label, counts)
                u_by_variant.setdefault(label, probabilities)
                results.append(row)
                print("U_BENCH_TRIAL " + json.dumps(row, sort_keys=True), flush=True)
            finally:
                module._MUCountsAccumulator = original
                logger.setLevel(old_level)
    expected = next(row for row in results if row["variant"] == "control")
    for row in results:
        assert row["counts_sha256"] == expected["counts_sha256"], row
        assert row["u_sha256"] == expected["u_sha256"], row
    summary = {
        label: {"median_seconds": statistics.median(
            r["wall_seconds"] for r in results if r["variant"] == label
        )} for label in variants
    }
    for label in summary:
        summary[label]["speedup"] = round(
            summary["control"]["median_seconds"]/summary[label]["median_seconds"], 3
        )
    result = {
        "seed": seed, "max_pairs": max_pairs, "num_chunks": num_chunks,
        "min_count_per_level": min_count_per_level, "repetitions": repetitions,
        "duckdb": duckdb.__version__, "pyarrow": pyarrow.__version__,
        "threads": linker._db_api.duckdb_con.sql("SELECT current_setting('threads')").fetchall()[0][0],
        "source_sha256": sources, "counts_and_u_identical": True,
        "trials": results, "summary": summary,
    }
    print("U_BENCH_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result
