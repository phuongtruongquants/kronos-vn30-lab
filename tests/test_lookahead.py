"""Regression checks for the inference boundary and sealed forward ledger."""

import json
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


if __name__ == "__main__":
    unittest.main()
