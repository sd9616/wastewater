# REPLACE YOUR EXISTING ARIMAX EVALUATION CODE WITH THIS:

import pandas as pd

def save_arimax_results_to_csv(results_list, filename):
    """Save ARIMAX model results to CSV file"""
    rows = []
    for combination, results in results_list:
        model_order = results['model'].order
        p, d, q = model_order
        metrics = results['metrics']
        
        row = {
            'exog_variables': ', '.join(combination) if combination else 'None',
            'arimax_p': p,
            'arimax_d': d, 
            'arimax_q': q,
            'arimax_order': f"ARIMAX({p}, {d}, {q})",
            'aic': metrics['aic'],
            'bic': metrics['bic'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'num_exog_vars': len(combination)
        }
        rows.append(row)
    
    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values('rmse')
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df_results

# Step 1: Run all combinations and collect results (no plots saved yet)
print("Running ARIMAX models for all combinations...")
spline_ww_combinations_results = []

for i, combination in enumerate(correlation_combinations_spline_ww):
    print(f"Processing combination {i+1}/{len(correlation_combinations_spline_ww)}: {combination}")
    
    model, forecast, results = seasonal(
        df=df_monthly_lagged_spline_ww,
        target_col='Spline_WW',
        exog_cols=combination,
        title_prefix=f"Monthly ARIMAX Model for Spline_WW with exog_vars = {combination}",
        auto=True
    )

    print(f"  ARIMAX{model.order} - RMSE: {results['metrics']['rmse']:.4f}, AIC: {results['metrics']['aic']:.4f}")
    spline_ww_combinations_results.append((combination, results))

# Step 2: Save all results to CSV
csv_filename = 'spline_ww_arimax_results_all.csv'
df_all_results = save_arimax_results_to_csv(spline_ww_combinations_results, csv_filename)

print(f"\nSummary of {len(spline_ww_combinations_results)} models:")
print(f"Best RMSE: {df_all_results['rmse'].min():.4f}")
print(f"Mean RMSE: {df_all_results['rmse'].mean():.4f}")
print(f"Best AIC: {df_all_results['aic'].min():.4f}")

# Step 3: Sort and show top 10
sorted_results_spline_ww = sorted(spline_ww_combinations_results, key=lambda x: x[1]['metrics']['rmse'])

print(f"\nTop 10 models by RMSE:")
for i, (exogs, result) in enumerate(sorted_results_spline_ww[:10]):
    model_order = result['model'].order
    print(f"{i+1:2d}. ARIMAX{model_order} - Exog: {exogs}, RMSE: {result['metrics']['rmse']:.4f}")

# Step 4: Re-run ONLY top 10 with plots
print(f"\nRe-running top 10 models with plot generation...")
top_10_results = []

for i, (combination, _) in enumerate(sorted_results_spline_ww[:10]):
    print(f"Re-running top model {i+1}: {combination}")
    
    model, forecast, results = seasonal(
        df=df_monthly_lagged_spline_ww,
        target_col='Spline_WW',
        exog_cols=combination,
        title_prefix=f"TOP {i+1} - Monthly ARIMAX Model for Spline_WW with exog_vars = {combination}",
        auto=True
    )
    
    top_10_results.append((combination, results))
    print(f"  Completed: ARIMAX{model.order} - RMSE: {results['metrics']['rmse']:.4f}")

# Step 5: Save top 10 results
top_10_csv_filename = 'spline_ww_arimax_top_10_results.csv'
df_top_10 = save_arimax_results_to_csv(top_10_results, top_10_csv_filename)

print(f"\nEvaluation complete!")
print(f"- All {len(spline_ww_combinations_results)} results: {csv_filename}")
print(f"- Top 10 results: {top_10_csv_filename}")
print(f"- Plots generated for top 10 models only")
