# Bank Profitability Forecasting Using Linear Regression

## Business question
How are deposit growth, loan growth, lending yields, deposit costs, fee income, operating expenses and credit impairment associated with monthly profit before tax?

## Data notice
`data/bank_profitability_demo.csv` is **synthetic demonstration data**. It is not sourced from Absa, GCB or any other bank. Results are not the user's original project results and must not replace them in a CV.

## Method
1. Load and validate monthly data.
2. Conduct exploratory analysis.
3. Use a chronological train/test split.
4. Fit a linear regression model.
5. Compare it with a persistence baseline.
6. Report R-squared, MAE and RMSE.

## Run
```bash
pip install -r requirements.txt
python src/forecast_profitability.py
```

## Important
Model performance is dataset-specific. Do not assume this demonstration run will reproduce previously reported metrics. For any CV metrics, use only the original dataset, code and test results that generated them.
