# Kronos Lab · VN30

Interactive one-stock-at-a-time Kronos-small forecasts for the 30 VN30 constituents in the research snapshot. Historical OHLCV data and a verified 2026-10-26 update are bundled for reproducibility. Prices are adjusted and shown in thousands of VND. This is a model demo, not a trading recommendation.

Entry point: `research/kronos_single_stock_web_20261027/app.py`

```bash
pip install -r requirements.txt
streamlit run research/kronos_single_stock_web_20261027/app.py
```

Kronos model source in `vendor/Kronos/model/` is copied from [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) at commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a` under its MIT license. Kronos-small weights and tokenizer are fetched from Hugging Face at pinned revisions on first inference. The 2026-10-26 OHLCV overlay comes from the Vietcap chart API; provenance and checks are in `manifest.json`.
