# Bank Profitability Driver Analysis

> **Portfolio case study · Python · pandas · scikit-learn · Matplotlib**

## Executive summary
This project investigates how bank income, funding costs, operating expenses and credit impairment relate to monthly profit before tax. It demonstrates a chronological model evaluation workflow and comparison with a simple persistence benchmark.

## Business question
How do net interest income, fee income, operating expenses, credit impairment, deposit growth, loan growth, lending yield and deposit cost relate to monthly profit before tax?

## Demonstration result
<image src="docs/profitability-metrics.svg" alt="Synthetic demonstration model evaluation metrics" />

| Metric | Demo result |
|---|---:|
| R-squared | 0.8096 |
| Mean absolute error | GHS 1.67m |
| Root mean squared error | GHS 1.98m |
| Persistence baseline MAE | GHS 4.15m |

The demonstration model improves on the persistence baseline MAE. These results come only from the synthetic CSV in this repository.

## Workflow
1. Validate and sort monthly records.
2. Use a chronological 80/20 train/test split.
3. Fit a linear regression using financial and business drivers.
4. Compare held-out estimates with a persistence baseline.
5. Export metrics, predictions, a comparison chart and model notes.

## Important modelling distinction
This implementation is a **driver-based estimate/nowcast**, not a fully ex-ante forecast. Net interest income and other accounting drivers must already be available as estimates at prediction time. A genuine forward forecast should separately forecast each driver using only information available before the target period.

## Run locally
```bash
pip install -r requirements.txt
python src/forecast_profitability.py
```

Generated files are written to `outputs/`.

## Repository structure
- `data/` — synthetic monthly demonstration dataset
- `src/` — analysis and modelling script
- `docs/` — portfolio chart
- `.github/workflows/` — automated validation

## Limitations and responsible use
The dataset is synthetic and does not represent Absa, GCB or any other bank. The metrics are not the user's original project results and should not be presented as real-bank performance. Model fit on synthetic data does not establish generalisation to live banking data. For a production model, validate data lineage, feature availability, out-of-time performance, residuals, stability and business assumptions.
