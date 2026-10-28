"""Regression checks for the inference boundary and sealed forward ledger."""

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


APP = Path(__file__).resolve().parents[1] / "research" / "kronos_single_stock_web_20261027"
sys.path.insert(0, str(APP))
import core  # noqa: E402
import forward  # noqa: E402
import retrospective  # noqa: E402


class LeakageBoundaryTest(unittest.TestCase):
    def test_inference_input_has_no_future_labels(self):
        prepared = core.prepare("FPT", "2026-10-27")
        self.assertFalse(hasattr(prepared, "actual"))
        self.assertEqual(prepared.history["date"].max(), "2026-10-27")
        self.assertEqual(len(prepared.history), 90)

    def test_future_calendar_cannot_change_model_timestamps(self):
        baseline = core.prepare("FPT", "2026-10-20").target_times.tolist()
        original = core.load_calendar()
        with patch.object(core, "load_calendar", return_value=[day for day in original if day != "2026-10-22"]):
            changed = core.prepare("FPT", "2026-10-20").target_times.tolist()
        self.assertEqual(baseline, changed)

    def test_predictor_receives_only_past_ohlcv(self):
        class RecordingPredictor:
            def predict(self, data, timestamps, target_times, **kwargs):
                self.latest_input = max(timestamps).date()
                self.first_target = min(target_times).date()
                return pd.DataFrame(np.ones((10, 5)), columns=["open", "high", "low", "close", "volume"])

        import threading
        candidate = RecordingPredictor()
        prepared = core.prepare("FPT", "2026-10-27")
        result = core.predict(prepared, candidate, threading.Lock())
        self.assertLess(candidate.latest_input, candidate.first_target)
        self.assertEqual(len(result), 10)

    def test_sealed_run_has_no_score_before_outcomes(self):
        run = forward.LEDGER / "2026-10-27"
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(len(manifest["symbols"]), 30)
        self.assertEqual(len(pd.read_csv(run / "forecasts.csv")), 900)
        self.assertTrue(forward.evaluate(run).empty)
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp)
            for name in ("manifest.json", "inputs.csv", "forecasts.csv"):
                (copy / name).write_bytes((run / name).read_bytes())
            with (copy / "forecasts.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                forward.evaluate(copy)

    def test_retrospective_recomputes_from_saved_paths_and_outcomes(self):
        run = retrospective.RUN
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["kind"], "exploratory_retrospective_not_point_in_time")
        for name, expected in manifest["file_sha256"].items():
            self.assertEqual(hashlib.sha256((run / f"{name}.csv").read_bytes()).hexdigest(), expected)
        inputs = pd.read_csv(run / "inputs.csv")
        forecasts = pd.read_csv(run / "forecasts.csv")
        outcomes = pd.read_csv(run / "outcomes.csv")
        summary = pd.read_csv(run / "summary.csv").set_index("seed")
        self.assertTrue((inputs.date <= inputs.cutoff).all())
        self.assertTrue((forecasts.cutoff >= "2025-07-01").all())
        self.assertEqual(len(outcomes), manifest["selected_pairs"])
        self.assertGreater(int(outcomes.calendar_mismatch.sum()), 0)
        self.assertEqual(len(outcomes) + manifest["excluded_pairs"],
                         len(manifest["anchors"]) * len(core.current_members()))
        self.assertFalse(pd.read_csv(run / "exclusions.csv").reason.eq("forecast_calendar_mismatch").any())
        terminal = forecasts.sort_values("date").groupby(["symbol", "cutoff", "seed"]).tail(1)
        joined = terminal.merge(outcomes[["symbol", "cutoff", "last_close", "actual_return"]],
                                on=["symbol", "cutoff"], validate="many_to_one")
        joined["hit"] = (np.sign(joined.close / joined.last_close - 1)
                         == np.sign(joined.actual_return)).astype(int)
        counts = joined.groupby("seed").hit.agg(["sum", "size"])
        for seed in manifest["seeds"]:
            self.assertEqual(int(counts.loc[seed, "sum"]), int(summary.loc[seed, "correct"]))
            self.assertEqual(int(counts.loc[seed, "size"]), int(summary.loc[seed, "samples"]))


if __name__ == "__main__":
    unittest.main()
