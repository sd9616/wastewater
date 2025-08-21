# Improved ARIMAX Model Evaluation with CSV Export and Top 10 Re-run

import pandas as pd
import os

def save_arimax_results_to_csv(results_list, filename):
    """
    Save ARIMAX model results to CSV file
    
    Parameters:
    results_list: List of tuples (combination, results)
    filename: Output CSV filename
    """
    rows = []
    for combination, results in results_list:
        # Extract model order (p, d, q)
        model_order = results['model'].order
        p, d, q = model_order
        
        # Extract metrics
        metrics = results['metrics']
        
        # Create row
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
    
    # Create DataFrame and save
    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values('rmse')  # Sort by RMSE
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df_results

def evaluate_arimax_models_efficiently(correlation_combinations, df, target_col, seasonal_func):
    """
    Efficiently evaluate ARIMAX models by:
    1. Running all combinations without saving plots
    2. Saving results to CSV
    3. Re-running only top 10 with plots
    
    Parameters:
    correlation_combinations: List of exogenous variable combinations
    df: DataFrame with data
    target_col: Target column name
    seasonal_func: The seasonal function to use for modeling
    """
    
    # Step 1: Run all combinations without plots
    print("Running ARIMAX models for all combinations...")
    all_results = []

    for i, combination in enumerate(correlation_combinations):
        print(f"Processing combination {i+1}/{len(correlation_combinations)}: {combination}")
        
        # Use seasonal function - note: you may need to modify seasonal function 
        # to have a parameter to skip plot saving for efficiency
        model, forecast, results = seasonal_func(
            df=df,
            target_col=target_col,
            exog_cols=combination,
            title_prefix=f"Monthly ARIMAX Model for {target_col} with exog_vars = {combination}",
            auto=True
        )

        # Access and display results
        print(f"  ARIMAX{model.order} - RMSE: {results['metrics']['rmse']:.4f}, AIC: {results['metrics']['aic']:.4f}")

        all_results.append((combination, results))

    # Step 2: Save all results to CSV
    csv_filename = f'{target_col.lower()}_arimax_results_all.csv'
    df_all_results = save_arimax_results_to_csv(all_results, csv_filename)

    # Display summary statistics
    print(f"\nSummary of {len(all_results)} models:")
    print(f"Best RMSE: {df_all_results['rmse'].min():.4f}")
    print(f"Worst RMSE: {df_all_results['rmse'].max():.4f}")
    print(f"Mean RMSE: {df_all_results['rmse'].mean():.4f}")
    print(f"Best AIC: {df_all_results['aic'].min():.4f}")
    print(f"Best BIC: {df_all_results['bic'].min():.4f}")

    # Step 3: Sort results by RMSE
    sorted_results = sorted(all_results, key=lambda x: x[1]['metrics']['rmse'])

    print(f"\nTop 10 models by RMSE:")
    for i, (exogs, result) in enumerate(sorted_results[:10]):
        model_order = result['model'].order
        print(f"{i+1:2d}. ARIMAX{model_order} - Exog Vars: {exogs}, RMSE: {result['metrics']['rmse']:.4f}, AIC: {result['metrics']['aic']:.4f}")

    # Step 4: Re-run top 10 models with plot saving
    print(f"\nRe-running top 10 models with plot generation...")
    top_10_results = []

    for i, (combination, _) in enumerate(sorted_results[:10]):
        print(f"Re-running top model {i+1}: {combination}")
        
        # Re-run with seasonal function (this will save plots)
        model, forecast, results = seasonal_func(
            df=df,
            target_col=target_col,
            exog_cols=combination,
            title_prefix=f"TOP {i+1} - Monthly ARIMAX Model for {target_col} with exog_vars = {combination}",
            auto=True
        )
        
        top_10_results.append((combination, results))
        print(f"  Completed: ARIMAX{model.order} - RMSE: {results['metrics']['rmse']:.4f}")

    # Step 5: Save top 10 results to separate CSV
    top_10_csv_filename = f'{target_col.lower()}_arimax_top_10_results.csv'
    df_top_10 = save_arimax_results_to_csv(top_10_results, top_10_csv_filename)

    print(f"\nEvaluation complete!")
    print(f"- All {len(all_results)} model results saved to: {csv_filename}")
    print(f"- Top 10 model results saved to: {top_10_csv_filename}")
    print(f"- Plots generated for top 10 models only")
    
    return all_results, top_10_results, df_all_results, df_top_10

# Example usage (replace with your actual code):
"""
# Your existing code setup should include:
# - correlation_combinations_spline_ww
# - df_monthly_lagged_spline_ww  
# - seasonal function

# Then call:
all_results, top_10_results, df_all, df_top10 = evaluate_arimax_models_efficiently(
    correlation_combinations=correlation_combinations_spline_ww,
    df=df_monthly_lagged_spline_ww,
    target_col='Spline_WW',
    seasonal_func=seasonal
)
"""