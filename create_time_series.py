"""
Create time series visualization for weather variables over Harris County.

This script generates a time series plot with multiple weather variables:
- tmean (surface temperature) - Orange
- twbs (wet bulb temperature) - Green  
- ppt (precipitation) - Blue (on secondary y-axis)
- hurs (relative humidity) - Red
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# Load the monthly COVID-weather data
df_monthly_covid_weather = pd.read_csv(
    'Harris_County_Monthly_PRISM_weighted_mean_weather_vars_COVID_2020_2023.csv',
    index_col=0,
    parse_dates=True
)

# Define variables and colors
variables = ['tmean', 'twbs', 'ppt', 'hurs']
colors = ['orange', 'green', 'blue', 'red']

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 4))

# Plot each variable
for i, var in enumerate(variables):
    color = colors[i]
    
    # For precipitation, use a secondary y-axis
    if var == 'ppt':
        twin1 = ax.twinx()
        axplot = twin1
        twin1.spines['right'].set_color(color)
        twin1.tick_params(axis='x', colors=color)
        twin1.tick_params(axis='y', colors=color)
        twin1.set_ylabel('Precipitation [mm]', color=color)
    else:
        axplot = ax
    
    # Plot line and scatter points
    df_monthly_covid_weather[var].plot(ax=axplot, color=color, alpha=0.7, label=var)
    axplot.scatter(df_monthly_covid_weather.index,
                   df_monthly_covid_weather[var],
                   color=color,
                   alpha=0.7,
                   s=30)

# Create custom legend with proper labels
labels = ['$T_{s}$', r'$T_{wb}$', r'$P$', r'$RH$']
legend_elements = [
    Line2D([0], [0], color=colors[0], label=labels[0]),
    Line2D([0], [0], color=colors[1], label=labels[1]),
    Line2D([0], [0], color=colors[2], label=labels[2]),
    Line2D([0], [0], color=colors[3], label=labels[3])
]

# Configure the plot
ax.legend(ncol=4, handles=legend_elements, loc='upper center', fancybox=True, framealpha=0.5)
ax.set_xlabel('')
ax.set_ylabel(r'Temperature [$^\circ$C] / Relative Humidity [%]')
ax.grid()
ax.set_title('Time Series over Harris County', fontsize=10)

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Save the figure
output_path = 'plots/monthly_tmean_twbs_ppt_hurs.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Time series plot saved to: {output_path}")

# Also show the plot
plt.show()
