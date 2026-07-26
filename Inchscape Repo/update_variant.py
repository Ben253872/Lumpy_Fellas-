import pandas as pd

df = pd.read_csv('results/tables/advanced_forecasts_all_datasets.csv')
print(f'Before: variants = {df["variant"].unique()}')
df['variant'] = 'all_sku_history'
print(f'After: variants = {df["variant"].unique()}')
df.to_csv('results/tables/advanced_forecasts_all_datasets.csv', index=False)
print('File updated successfully')
