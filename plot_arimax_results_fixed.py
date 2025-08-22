import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima.arima.utils import ADFTest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

def plot_arimax_results(df, target_col, exog_cols, train_size=None,
                       figsize=(10, 5), title_prefix="ARIMAX Model", auto=True, order=(0, 0, 0), save=True, show=True, plots_folder_path="./plots"):
    """
    Comprehensive plotting function for ARIMAX model results

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    target_col : str
        Name of the target variable column
    exog_cols : list
        List of exogenous variable column names
    train_size : int, optional
        Number of observations for training. If None, uses 80% of data
    figsize : tuple
        Figure size for the plots
    title_prefix : str
        Prefix for plot titles
    auto : bool
        Whether to use auto_arima or manual ARIMA
    order : tuple
        ARIMA order (p, d, q) for manual mode
    save : bool
        Whether to save the plot
    show : bool
        Whether to display verbose output
    plots_folder_path : str
        Path to save plots

    Returns:
    --------
    model : fitted ARIMAX model
    forecast : forecast values
    results_dict : dictionary with model metrics and data
    """

    # Split
    if train_size is None:
        train_size = int(len(df) * 0.8)

    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    if show: 
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

    print("Creating ARIMAX model using", target_col, "and", *exog_cols)
    
    if auto:
        # Fit ARIMAX model
        model = pm.auto_arima(
            train_data[target_col],
            X=train_data[exog_cols],
            # test='adf',
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            max_p=3, max_q=3, max_d=2
        )
        y = train_data[target_col]
        adf_test = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test.should_diff(y)  # (0.01, False)

        if show: 
            print(f"p_val {p_val}, should_diff {should_diff}")
            print(f"")
            print(f"\nBest model: ARIMA{model.order}")
            print(f"AIC: {model.aic():.2f} | BIC: {model.bic():.2f}")

        # Forecast
        forecast, conf_int = model.predict(
            n_periods=len(test_data),
            X=test_data[exog_cols],
            return_conf_int=True
        )

        if show:
            print("forecast", forecast)
            print("test_data exog_cols shape:", test_data[exog_cols].shape)
        
    else:
        model = ARIMA(
            train_data[target_col],
            exog=train_data[exog_cols],
            order=order)
        # results = model.results()
        model_fit = model.fit()
        summary = model_fit.summary()
        
        if show:
            print(f"AIC: {model_fit.aic} | BIC: {model_fit.bic}")
            print(f"\nModel: ARIMA{order}")

        forecast = model_fit.predict(
            start=len(train_data),  # or results.nobs
            end=len(train_data) + len(test_data) - 1,
            exog=test_data[exog_cols]
        )
        
        if show:
            print("forecast", forecast)
            print("test_data exog_cols shape:", test_data[exog_cols].shape)

    # Combine for continuity
    last_train_point = train_data[target_col].iloc[-1:]
    combined_forecast = pd.concat([last_train_point, pd.Series(forecast, index=test_data.index)])
    combined_ground_truth = pd.concat([last_train_point, test_data[target_col]])

    if show:
        print("last_train_point", last_train_point) 
        print("combined_forecast", combined_forecast) 
        print("combined_ground_truth", combined_ground_truth) 
    
    # === Plot 1: Forecast Time Series ===
    plt.figure(figsize=figsize)
    plt.plot(train_data.index, train_data[target_col], label='Training Data', color='black')
    plt.plot(combined_ground_truth.index, combined_ground_truth, label='Ground Truth', color='green', linestyle='--', marker='o')
    plt.plot(combined_forecast.index, combined_forecast, label='Forecast', color='orange', linestyle='--', marker='x')

    # Generate title based on mode
    if auto:
        title = f"{title_prefix}, order {model.order}"
    else:
        title = f"{title_prefix}, order {order}"
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        # Create plots directory if it doesn't exist
        os.makedirs(plots_folder_path, exist_ok=True)
        filename = f'{title}.png'
        saved_plot_path = os.path.join(plots_folder_path, filename)
        plt.savefig(saved_plot_path)
    else:
        saved_plot_path = None

    if show: 
        plt.show()

    # Calculate metrics
    mse = mean_squared_error(test_data[target_col], forecast)
    mae = mean_absolute_error(test_data[target_col], forecast)
    rmse = np.sqrt(mse)

    if show: 
        # Plot exogenous variables
        plt.figure(figsize=figsize)
        for col in exog_cols:
            df[col].plot(label=col, linewidth=1.5)
        plt.title("Exogenous Variable Trends")
        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print model summary
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    # Prepare results dictionary
    if auto:
        results_dict = {
            'model': model,
            'forecast': forecast,
            'confidence_intervals': conf_int,
            'train_data': train_data,
            'test_data': test_data,
            'metrics': {
                'aic': model.aic(),
                'bic': model.bic(),
                'rmse': rmse,
                'mae': mae
            }
        }
    else:
        results_dict = {
            'model': model_fit,  # Return the fitted model for consistency
            'forecast': forecast,
            'train_data': train_data,
            'test_data': test_data,
            'metrics': {
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'rmse': rmse,
                'mae': mae
            }
        }

    if save and saved_plot_path: 
        results_dict['plot_path'] = saved_plot_path

    return model if auto else model_fit, forecast, results_dict