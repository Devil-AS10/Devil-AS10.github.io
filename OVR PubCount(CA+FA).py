#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Load the provided CSV file to check its structure and find the relevant columns for CA and FA differentiation
import pandas as pd
from sklearn.metrics import r2_score
from numpy import polyval

# Path to the dataset
data_path = 'Final Data CA+FA.csv'

# Load the dataset
data = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
data.head()


# In[7]:


# Filter the dataset for CA and FA
ca_data = data[data['Author Type'] == 'Corresponding']
fa_data = data[data['Author Type'] == 'First Author']

# Function to process and forecast publications
def forecast_publications(subset_data, author_type):
    # Count unique WOS ID occurrences per year
    publication_counts = subset_data.groupby('Publication Year')['WOS ID'].nunique()

    # Convert the series to a DataFrame
    publication_counts_df = publication_counts.reset_index()
    publication_counts_df.columns = ['Year', 'Publications']

    # Using numpy for polynomial fitting
    from numpy import polyfit, polyval, array
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Filter data for pre-pandemic years for modeling
    pre_pandemic_data = publication_counts_df[publication_counts_df['Year'] <= 2019]

    # Prepare data for modeling
    years = pre_pandemic_data['Year'].values
    publications = pre_pandemic_data['Publications'].values

    # Polynomial degree (quadratic model, degree=2)
    degree = 2

    # Fit the model
    publication_model = polyfit(years, publications, degree)

    # Years for forecasting
    forecast_years = np.array([2019, 2020, 2021, 2022, 2023])

    # Forecasting
    publication_forecasts = polyval(publication_model, forecast_years)

    # Actual data for comparison (we include 2020 since it's part of the forecast now)
    actual_publications = publication_counts_df[publication_counts_df['Year'].isin(forecast_years)]['Publications'].values

    # Prepare a DataFrame for manuscript-ready table
    forecast_vs_actual = pd.DataFrame({
        'Year': forecast_years,
        'Forecasted Publications': publication_forecasts.round(0),
        'Actual Publications': actual_publications
    })

    # Calculating RMSE and R-squared for the degree 2 model
    forecast_pre_pandemic = polyval(publication_model, years)
    rmse_pre_pandemic = np.sqrt(mean_squared_error(publications, forecast_pre_pandemic))
    r_squared_pre_pandemic = r2_score(publications, forecast_pre_pandemic)

    forecast_actual_period = polyval(publication_model, forecast_years)
    rmse_actual_period = np.sqrt(mean_squared_error(actual_publications, forecast_actual_period))
    r_squared_actual_period = r2_score(actual_publications, forecast_actual_period)

    metrics = (rmse_pre_pandemic, r_squared_pre_pandemic, rmse_actual_period, r_squared_actual_period)
    
    return forecast_vs_actual, metrics, publication_model
# Check the unique values in the 'Author Type' column to ensure correct labels are used for filtering
unique_author_types = data['Author Type'].unique()
unique_author_types
# Correct the filtering condition for First Author (FA) publications
fa_data = data[data['Author Type'] == 'First']

# Re-run the forecast function for both CA and FA with the correct filtering condition

# Forecast for CA
ca_forecast, ca_metrics, ca_model = forecast_publications(ca_data, 'CA')

# Forecast for FA
fa_forecast, fa_metrics, fa_model = forecast_publications(fa_data, 'FA')

(ca_forecast, ca_metrics, ca_model), (fa_forecast, fa_metrics, fa_model)



# In[8]:


def format_output(forecast, metrics, model, author_type):
    a, b, c = model
    output = f"{author_type} Publications Forecast\n"
    output += "The forecasted versus actual publications for the years 2019 through 2023 are as follows:\n"
    for index, row in forecast.iterrows():
        output += f"{int(row['Year'])}: Forecasted = {int(row['Forecasted Publications'])}, Actual = {int(row['Actual Publications'])}\n"
    output += "Model Metrics:\n"
    output += f"RMSE (Pre-pandemic): {metrics[0]:.2f}\n"
    output += f"R^2 (Pre-pandemic): {metrics[1]:.3f}\n"
    output += f"RMSE (Forecast period): {metrics[2]:.2f}\n"
    output += f"R^2 (Forecast period): {metrics[3]:.3f}\n"
    output += "Model Coefficients:\n"
    output += f"{a:.4f}x^2 - {abs(b):.5f}x + {c:.2f}\n"

    return output

# Output the formatted results for CA and FA
ca_output = format_output(ca_forecast, ca_metrics, ca_model, "Corresponding Author (CA)")
fa_output = format_output(fa_forecast, fa_metrics, fa_model, "First Author (FA)")

print(ca_output)
print(fa_output)

# Predict and return values for all years in the dataset
def predict_all_years(data, model):
    years = data['Publication Year'].unique()
    years.sort()
    predictions = polyval(model, years)
    return years, predictions.round(0)

ca_years, ca_predictions = predict_all_years(data[data['Author Type'] == 'Corresponding'], ca_model)
fa_years, fa_predictions = predict_all_years(data[data['Author Type'] == 'First'], fa_model)

ca_years, ca_predictions, fa_years, fa_predictions


# In[9]:


import matplotlib.pyplot as plt

# Re-plotting the forecasted publications for CA and FA along with actual data for comparison

# Setting up the plots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
axes[0].set_title('Corresponding Author (CA) Publications Forecast')
axes[1].set_title('First Author (FA) Publications Forecast')

# Corresponding Author (CA) Plot
axes[0].plot(ca_years, ca_predictions, label='Forecasted CA Publications', color='blue')
ca_actual = data[data['Author Type'] == 'Corresponding'].groupby('Publication Year')['WOS ID'].nunique()
axes[0].scatter(ca_actual.index, ca_actual, color='red', label='Actual CA Publications')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Publications')
axes[0].legend()
axes[0].grid(True)

# First Author (FA) Plot
axes[1].plot(fa_years, fa_predictions, label='Forecasted FA Publications', color='green')
fa_actual = data[data['Author Type'] == 'First'].groupby('Publication Year')['WOS ID'].nunique()
axes[1].scatter(fa_actual.index, fa_actual, color='orange', label='Actual FA Publications')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Publications')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# In[ ]:




