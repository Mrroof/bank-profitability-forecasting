# Bank Profitability Driver Analysis Using Linear Regression

## Business question
How are net interest income, fee income, operating expenses, credit impairment and key balance-sheet/pricing indicators associated with monthly profit before tax?

## Data notice
`data/bank_profitability_demo.csv` is **synthetic demonstration data**. It is not sourced from Absa, GCB or any other bank. Results are not the user's original project results and must not replace them in a CV.

## Method
1. Validate and sort the monthly observations.
2. Use a chronological 80/20 train/test split.
3. Fit a linear regression using accounting and business drivers.
4. Compare the estimates with a persistence baseline.
5. Report R-squared, MAE and RMSE.

## Important modelling distinction
The revised demonstration model is a **driver-based estimate/nowcast**, not a fully ex-ante forecast: net interest income and other accounting drivers must be available as forecasts or estimates at prediction time. A production forecast should forecast those drivers separately using information available before the target month.

## Run
```bash
pip install -r requirements.txt
python src/forecast_profitability.py
```

## Outputs
Metrics, held-out predictions, an actual-versus-estimated chart and model notes are written to `outputs/`.

## Limitations
Model performance is dataset-specific. Do not assume this demonstration run will reproduce previously reported metrics. For any CV metrics, use only the original dataset, code and test results that generated them.
