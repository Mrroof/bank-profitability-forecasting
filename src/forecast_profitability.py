from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/bank_profitability_demo.csv", parse_dates=["month"])
df = df.sort_values("month").reset_index(drop=True)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

target = "profit_before_tax_ghs"
drivers = [
    "net_interest_income_ghs", "fee_income_ghs",
    "operating_expenses_ghs", "credit_impairment_ghs",
    "deposit_growth_rate", "loan_growth_rate",
    "lending_yield", "deposit_cost"
]
required = [target, "month", *drivers]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
df = df.dropna(subset=required).reset_index(drop=True)
if len(df) < 20:
    raise ValueError("At least 20 complete monthly observations are required")

# Model A: contemporaneous driver-based estimate/nowcast. This is useful when
# the current month's accounting drivers are already known or estimated.
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
nowcast = LinearRegression().fit(train[drivers], train[target])
nowcast_pred = nowcast.predict(test[drivers])

# Model B: genuine one-month-ahead baseline forecast. Every feature is shifted
# by one month, so month t is predicted using information available through t-1.
lagged = df[["month", target, *drivers]].copy()
lag_features = []
for column in [target, *drivers]:
    name = f"{column}_lag1"
    lagged[name] = lagged[column].shift(1)
    lag_features.append(name)
lagged = lagged.dropna().reset_index(drop=True)

lag_split = int(len(lagged) * 0.8)
lag_train, lag_test = lagged.iloc[:lag_split], lagged.iloc[lag_split:]
forecast = LinearRegression().fit(lag_train[lag_features], lag_train[target])
forecast_pred = forecast.predict(lag_test[lag_features])
persistence_pred = lag_test[f"{target}_lag1"].to_numpy()

def metric_row(actual, predicted, baseline, model_name, train_rows, test_rows):
    return {
        "model": model_name,
        "R2": r2_score(actual, predicted),
        "MAE_GHS": mean_absolute_error(actual, predicted),
        "RMSE_GHS": mean_squared_error(actual, predicted) ** 0.5,
        "Persistence_MAE_GHS": mean_absolute_error(actual, baseline),
        "Train_rows": train_rows,
        "Test_rows": test_rows
    }

nowcast_persistence = test[target].shift(1)
nowcast_persistence.iloc[0] = train[target].iloc[-1]
metrics = pd.DataFrame([
    metric_row(test[target], nowcast_pred, nowcast_persistence,
               "Contemporaneous driver nowcast", len(train), len(test)),
    metric_row(lag_test[target], forecast_pred, persistence_pred,
               "One-month-ahead lagged-driver forecast", len(lag_train), len(lag_test))
])
metrics.to_csv(out / "model_metrics.csv", index=False)

predictions = pd.DataFrame({
    "forecast_month": lag_test["month"],
    "actual_profit": lag_test[target].to_numpy(),
    "one_month_ahead_forecast": forecast_pred,
    "persistence_prediction": persistence_pred
})
predictions.to_csv(out / "ex_ante_forecast_predictions.csv", index=False)

chart = pd.DataFrame({
    "month": lag_test["month"].to_numpy(),
    "actual_profit": lag_test[target].to_numpy(),
    "one_month_ahead_forecast": forecast_pred,
    "persistence_prediction": persistence_pred
})
ax = chart.plot(
    x="month", y=["actual_profit", "one_month_ahead_forecast", "persistence_prediction"],
    figsize=(10, 4), title="One-month-ahead demo forecast vs actual and persistence"
)
ax.set_ylabel("Profit before tax (GHS)")
plt.tight_layout()
plt.savefig(out / "ex_ante_forecast.png", dpi=160)
plt.close()

(out / "model_notes.txt").write_text(
    "SYNTHETIC DEMONSTRATION DATA ONLY\n"
    "The contemporaneous model is a driver-based estimate/nowcast and uses "
    "same-month accounting drivers.\n"
    "The one-month-ahead model uses only prior-month profit and financial "
    "drivers; it is evaluated chronologically against a persistence baseline.\n"
    "The small synthetic sample is illustrative, not evidence of live-bank "
    "forecast performance. Review residuals, assumptions, feature availability "
    "and out-of-time stability before any practical use.\n",
    encoding="utf-8"
)

print("SYNTHETIC DEMONSTRATION DATA ONLY")
print(metrics.to_string(index=False))
print("\nOne-month-ahead model coefficients:")
print(pd.Series(forecast.coef_, index=lag_features).sort_values(ascending=False))
