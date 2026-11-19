from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/bank_profitability_demo.csv",parse_dates=["month"]).sort_values("month")
out=ROOT/"outputs"; out.mkdir(exist_ok=True)
target="profit_before_tax_ghs"
features=["deposit_growth","loan_growth","lending_yield","deposit_cost",
"fee_income_ghs","operating_expenses_ghs","credit_impairment_ghs"]
split=int(len(df)*.8); train,test=df.iloc[:split],df.iloc[split:]
model=LinearRegression().fit(train[features],train[target])
pred=model.predict(test[features])
baseline=test[target].shift(1); baseline.iloc[0]=train[target].iloc[-1]
metrics={"R2":r2_score(test[target],pred),
"MAE_GHS":mean_absolute_error(test[target],pred),
"RMSE_GHS":mean_squared_error(test[target],pred)**.5,
"Persistence_MAE_GHS":mean_absolute_error(test[target],baseline)}
pd.DataFrame([metrics]).to_csv(out/"model_metrics.csv",index=False)
pd.DataFrame({"month":test.month,"actual_profit":test[target],"predicted_profit":pred,
"persistence_prediction":baseline.values}).to_csv(out/"test_predictions.csv",index=False)
ax=test.assign(predicted=pred).plot(x="month",y=[target,"predicted"],figsize=(9,4),
 title="Demo test period: actual vs predicted profit")
ax.set_ylabel("Profit before tax (GHS)"); plt.tight_layout()
plt.savefig(out/"actual_vs_predicted.png",dpi=160); plt.close()
print("SYNTHETIC DEMONSTRATION DATA ONLY")
for k,v in metrics.items(): print(f"{k}: {v:,.4f}")
print("\nCoefficients:")
print(pd.Series(model.coef_,index=features).sort_values(ascending=False))
