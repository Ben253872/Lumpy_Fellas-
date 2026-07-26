import pandas as pd

original = pd.read_csv("Inchscape Repo/results/tables/rolling_evaluation_erratic_chronos_erratic_features_plus_external_monthly.csv")
regenerated = pd.read_csv("Inchscape Repo/results/tables/erratic_chronos2_external_monthly_aligned.csv")

print("Comparing original vs regenerated monthly WMAPE:")
print("Month Original Regenerated Difference")
print("-" * 45)

total_orig_err = 0
total_regen_err = 0
matches = 0

for idx, orig_row in original.iterrows():
    month_str = orig_row['eval_month']
    regen_row = regenerated[regenerated['eval_month'] == month_str]
    
    if len(regen_row) > 0:
        orig_wmape = float(orig_row['wmape'])
        regen_wmape = float(regen_row.iloc[0]['wmape'])
        diff = regen_wmape - orig_wmape
        
        if abs(diff) < 0.01:
            matches += 1
            mark = " ✓"
        else:
            mark = ""
        
        print(f"{month_str} {orig_wmape:7.2f}    {regen_wmape:7.2f}      {diff:+7.2f}{mark}")
        total_orig_err += orig_wmape
        total_regen_err += regen_wmape

print("-" * 45)
print(f"Mean:  {total_orig_err/19:7.2f}    {total_regen_err/19:7.2f}      {(total_regen_err-total_orig_err)/19:+7.2f}")
print(f"\nExact matches (diff < 0.01): {matches}/19")
