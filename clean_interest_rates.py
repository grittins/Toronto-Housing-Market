import pandas as pd

filetoload = "Resources/mortgage_rates.csv"

df = pd.read_csv(filetoload)
#print(df.head(10))

avg_quarters_df = df.groupby(["Year_Quarter_Key"]).mean()["five_year_Mortgage_Rates"]

#avg_quarters_df.to_csv(index=False)

from pathlib import Path  
filepath = Path('Resources/clean_rates.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
avg_quarters_df.to_csv(filepath)