# Kronos Lab · VN30

Interactive one-stock-at-a-time Kronos-small forecasts for the 30 VN30 constituents in the research snapshot. Historical OHLCV data and verified updates through **2026-10-27** are bundled for reproducibility. Prices are adjusted and shown in thousands of VND. This is a model demo, not a trading recommendation.

The app displays 1, 3, or 5 independently sampled paths (one fixed seed per path), a price and volume chart with zoom and pan, and a descriptive seed audit. The audit uses a fixed non-overlapping 2026 schedule: 10 windows with 10-session horizon, six for calibration and four later windows for a small holdout. It avoids future bars in prediction inputs but cannot establish full out-of-sample status because the checkpoint training cutoff and historical data vintages are not verified.

Entrypoint: `research/kronos_single_stock_web_20261027/app.py`

```bash
pip install -r requirements.txt
streamlit run research/kronos_single_stock_web_20261027/app.py
```

Kronos model source in `vendor/Kronos/model/` is copied from [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) at commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a` under its MIT license. Kronos-small weights and tokenizer are fetched from Hugging Face at pinned revisions on first inference. The 2026-10-26 and 2026-10-27 OHLCV overlays come from the Vietcap chart API; provenance and overlap checks are in their manifests.
