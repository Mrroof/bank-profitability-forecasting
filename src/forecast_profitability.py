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
# Accounting drivers are included because the demo target is constructed largely
# from net interest income, fees, operating expenses and impairment.
features = [
    "net_interest_income_ghs", "fee_income_ghs",
    "operating_expenses_ghs", "credit_impairment_ghs",
    "deposit_growth_rate", "loan_growth_rate",
    "lending_yield", "deposit_cost"
]
missing = [c for c in [target, "month", *features] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
df = df.dropna(subset=[target, *features]).reset_index(drop=True)
if len(df) < 20:
    raise ValueError("At least 20 complete monthly observations are required")

split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
model = LinearRegression().fit(train[features], train[target])
pred = model.predict(test[features])

baseline = test[target].shift(1)
baseline.iloc[0] = train[target].iloc[-1]
metrics = {
    "R2": r2_score(test[target], pred),
    "MAE_GHS": mean_absolute_error(test[target], pred),
    "RMSE_GHS": mean_squared_error(test[target], pred) ** 0.5,
    "Persistence_MAE_GHS": mean_absolute_error(test[target], baseline),
    "Train_rows": len(train),
    "Test_rows": len(test)
}
pd.DataFrame([metrics]).to_csv(out / "model_metrics.csv", index=False)
pd.DataFrame({
    "month": test.month,
    "actual_profit": test[target],
    "predicted_profit": pred,
    "persistence_prediction": baseline.values
}).to_csv(out / "test_predictions.csv", index=False)

ax = test.assign(predicted=pred).plot(
    x="month", y=[target, "predicted"], figsize=(9, 4),
    title="Demo test period: actual vs driver-based estimate"
)
ax.set_ylabel("Profit before tax (GHS)")
plt.tight_layout()
plt.savefig(out / "actual_vs_predicted.png", dpi=160)
plt.close()

(out / "model_notes.txt").write_text(
    "SYNTHETIC DEMONSTRATION DATA ONLY\n"
    "Chronological 80/20 holdout. This is a driver-based estimate/nowcast, "
    "not a fully ex-ante forecast: net interest income and other accounting "
    "drivers must be available as forecasts or estimates at prediction time.\n"
    "Do not present these demo metrics as results from live bank data.\n",
    encoding="utf-8"
)
print("SYNTHETIC DEMONSTRATION DATA ONLY")
for key, value in metrics.items():
    print(f"{key}: {value:,.4f}")
print("\nCoefficients:")
print(pd.Series(model.coef_, index=features).sort_values(ascending=False))
