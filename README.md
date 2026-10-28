# Kronos Lab · VN30

Interactive one-stock-at-a-time Kronos-small forecasts for the 30 VN30 constituents in the research snapshot plus **DGW as a separately labeled extra stock**. Historical OHLCV data through **2026-10-27** are bundled for reproducibility. Prices are shown in thousands of VND. This is a model demo, not a trading recommendation.

The app displays 1, 3, or 5 independently sampled paths (one fixed seed per path), an interactive price and volume chart with predicted OHLC candles, and two separate evaluations. The exploratory retrospective replays fixed 10-session forecasts from July 2025 on the 30-stock snapshot and compares them with later prices and simple baselines. It does not establish point-in-time out-of-sample performance: historical price vintages and historical index membership are unavailable. The forward ledger seals forecasts before their results exist and will only report a win rate after all 10 future sessions are observed.

DGW uses a separate VCI price snapshot and a separate 33-window retrospective with the same seed and horizon settings. It is excluded from the 30-stock aggregate and from the sealed forward ledger. “Always up” predicts a positive 10-session return at every cutoff; “10-session momentum” predicts the next 10-session direction will match the previous 10-session direction. These are comparison rules, not the five sample paths on the chart.

Entrypoint: `research/kronos_single_stock_web_20261027/app.py`

```bash
pip install -r requirements.txt
streamlit run research/kronos_single_stock_web_20261027/app.py
```

Kronos model source in `vendor/Kronos/model/` is copied from [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) at commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a` under its MIT license. Kronos-small weights and tokenizer are fetched from Hugging Face at pinned revisions on first inference. The 2026-10-26 and 2026-10-27 OHLCV overlays come from the Vietcap chart API; provenance and overlap checks are in their manifests.
