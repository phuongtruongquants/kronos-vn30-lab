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
import extra_replay  # noqa: E402


class LeakageBoundaryTest(unittest.TestCase):
    def test_latest_bars_and_default_temperature(self):
        self.assertEqual(core.T, 1.0)
        for date in ("2026-10-28", "2026-10-29"):
            directory = core.HERE / f"data_update_{date.replace('-', '')}"
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["target_session"], date)
            self.assertEqual(manifest["rows"], 31)
            self.assertEqual(set(manifest["symbols"]), set(core.current_members()) | {"DGW"})
            self.assertEqual(hashlib.sha256((directory / f"ohlcv_{date}.csv").read_bytes()).hexdigest(), manifest["sha256"])
        for symbol in manifest["symbols"]:
            self.assertEqual(core.load_stock(symbol)[0].index[-1], "2026-10-29")
        tcb = next(check for check in manifest["checks"] if check["symbol"] == "TCB")
        self.assertEqual(tcb["overlap_price_divisor"], 1.5)
        self.assertTrue(tcb["overlap_volume_verified"])
        self.assertEqual(manifest["corporate_actions"][0]["symbol"], "TCB")

    def test_tcb_stock_dividend_keeps_forecasts_on_their_cutoff_basis(self):
        raw, _ = core.load_stock("TCB")
        before = core.prepare("TCB", "2026-10-28")
        after = core.prepare("TCB", "2026-10-29")
        self.assertAlmostEqual(float(before.history.loc["2026-10-28", "close"]),
                               float(raw.loc["2026-10-28", "close"]))
        self.assertAlmostEqual(float(after.history.loc["2026-10-28", "close"]),
                               float(raw.loc["2026-10-28", "close"]) / 1.5)
        self.assertAlmostEqual(float(after.history.loc["2026-10-29", "close"]),
                               float(raw.loc["2026-10-29", "close"]))
        realized = core.realized_path("TCB", "2026-10-28")
        self.assertAlmostEqual(float(realized.loc["2026-10-29", "close"]),
                               float(raw.loc["2026-10-29", "close"]) * 1.5)

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
        if all(len(core.realized_path(symbol, manifest["cutoff"])) < manifest["horizon"]
               for symbol in manifest["symbols"]):
            self.assertTrue(forward.evaluate(run).empty)
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp)
            for name in ("manifest.json", "inputs.csv", "forecasts.csv"):
                (copy / name).write_bytes((run / name).read_bytes())
            with (copy / "forecasts.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                forward.evaluate(copy)

    def test_october_28_temperature_one_run_is_sealed_without_outcomes(self):
        run = forward.LEDGER / "2026-10-28"
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["temperature"], 1.0)
        self.assertEqual(manifest["cutoff"], "2026-10-28")
        self.assertEqual(len(manifest["symbols"]), 30)
        forecasts = pd.read_csv(run / "forecasts.csv")
        self.assertEqual(len(forecasts), 900)
        self.assertTrue(forecasts.date.gt(manifest["cutoff"]).all())
        self.assertFalse((run / "outcomes.csv").exists())

    def test_october_29_run_uses_adjusted_tcb_input_and_no_outcomes(self):
        run = forward.LEDGER / "2026-10-29"
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["temperature"], 1.0)
        self.assertEqual(len(manifest["symbols"]), 30)
        inputs = pd.read_csv(run / "inputs.csv")
        tcb = inputs.loc[inputs.symbol.eq("TCB")].set_index("date")
        self.assertAlmostEqual(float(tcb.loc["2026-10-28", "close"]), 33.5 / 1.5)
        self.assertAlmostEqual(float(tcb.loc["2026-10-29", "close"]), 22.35)
        self.assertFalse((run / "outcomes.csv").exists())

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

    def test_dgw_replay_is_separate_and_recomputable(self):
        self.assertEqual(len(core.current_members()), 30)
        self.assertNotIn("DGW", core.current_members())
        frame, metadata = core.load_stock("DGW")
        self.assertEqual(frame.index[-1], "2026-10-29")
        self.assertEqual(len(frame), metadata["rows"] + 2)
        data_path = core.EXTRA_DATA_ROOT / "DGW.csv"
        self.assertEqual(hashlib.sha256(data_path.read_bytes()).hexdigest(), metadata["sha256"])
        self.assertEqual(len(core.prepare("DGW", "2026-10-27").history), 90)
        run = extra_replay.RUN
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["symbol"], "DGW")
        self.assertEqual(manifest["data_sha256"], metadata["sha256"])
        for name, digest in manifest["file_sha256"].items():
            self.assertEqual(hashlib.sha256((run / f"{name}.csv").read_bytes()).hexdigest(), digest)
        inputs = pd.read_csv(run / "inputs.csv")
        self.assertTrue((inputs.date <= inputs.cutoff).all())
        forecasts = pd.read_csv(run / "forecasts.csv")
        outcomes = pd.read_csv(run / "outcomes.csv")
        summary = pd.read_csv(run / "summary.csv").set_index("seed")
        terminal = forecasts.sort_values("date").groupby(["cutoff", "seed"]).tail(1)
        joined = terminal.merge(outcomes[["cutoff", "last_close", "actual_return"]],
                                on="cutoff", validate="many_to_one")
        hits = (np.sign(joined.close / joined.last_close - 1)
                == np.sign(joined.actual_return)).astype(int)
        for seed in manifest["seeds"]:
            subset = joined.seed.eq(seed)
            self.assertEqual(int(hits[subset].sum()), int(summary.loc[seed, "correct"]))
            self.assertEqual(int(subset.sum()), int(summary.loc[seed, "samples"]))

    def test_temperature_replays_are_paired_without_new_labels(self):
        for baseline in (retrospective.RUN, extra_replay.RUN):
            candidate = baseline.with_name(baseline.name + "_T1")
            old = json.loads((baseline / "manifest.json").read_text())
            new = json.loads((candidate / "manifest.json").read_text())
            self.assertEqual(old["temperature"], 0.6)
            self.assertEqual(new["temperature"], 1.0)
            for key in ("seeds", "top_p", "lookback", "horizon", "model_sha256", "tokenizer_sha256"):
                self.assertEqual(old[key], new[key])
            for name in ("inputs", "outcomes", "exclusions"):
                self.assertEqual(old["file_sha256"][name], new["file_sha256"][name])
            if baseline == retrospective.RUN:
                self.assertEqual(old["file_sha256"]["plan"], new["file_sha256"]["plan"])


if __name__ == "__main__":
    unittest.main()
