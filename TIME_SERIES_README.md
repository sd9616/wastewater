# Time Series Visualization Script

## Overview
This script (`create_time_series.py`) generates a time series visualization of weather variables over Harris County from 2020-2023.

## Variables Plotted
- **tmean** (Surface Temperature, Ts) - Orange line
- **twbs** (Wet Bulb Temperature, Twb) - Green line
- **ppt** (Precipitation, P) - Blue line (on secondary y-axis)
- **hurs** (Relative Humidity, RH) - Red line

## Usage
```bash
python3 create_time_series.py
```

## Requirements
- pandas
- matplotlib
- Harris_County_Monthly_PRISM_weighted_mean_weather_vars_COVID_2020_2023.csv

## Output
The script generates a high-resolution plot saved to:
- `plots/monthly_tmean_twbs_ppt_hurs.png`

## Key Features
- Dual y-axes: left for temperature/humidity, right for precipitation
- Both line plots and scatter markers for each variable
- Custom legend with mathematical notation
- Grid for easy reading of values
- Time series spans January 2020 to December 2023

## Fixed Issues
This script addresses several issues from the original code:
1. Fixed incorrect `.plot()` method calls with wrong x parameter
2. Fixed `.scatter()` method to use axes scatter directly
3. Added missing `Line2D` import
4. Added 4th legend label for relative humidity
5. Corrected y-axis label to include both temperature and humidity units
