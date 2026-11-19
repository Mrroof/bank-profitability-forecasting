# Sample dataset

`bank_profitability_demo.csv.gz` is a compressed CSV sample containing 72 monthly observations for a bank profitability forecasting demonstration. It includes deposit and loan balances and growth, lending yield, deposit cost, fee income, operating expenses, credit impairment, net interest income, and profit before tax.

The values are illustrative demonstration data, not actual customer or bank records. Load directly with pandas:

```python
import pandas as pd

df = pd.read_csv("data/bank_profitability_demo.csv.gz", compression="gzip")
```
