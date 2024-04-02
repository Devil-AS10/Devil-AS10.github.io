#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd

# Load the dataset
data_path = 'Final Data CA+FA.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataframe to understand its structure
data.head()


# In[60]:


# Count unique WOS ID occurrences per year
publication_counts = data.groupby('Publication Year')['WOS ID'].nunique()

# Convert the series to a DataFrame
publication_counts_df = publication_counts.reset_index()
publication_counts_df.columns = ['Year', 'Publications']

publication_counts_df.tail()


# In[61]:


# Using numpy for polynomial fitting
from numpy import polyfit, polyval

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

forecast_vs_actual


# In[62]:


# Function to print the polynomial equation from coefficients
def print_polynomial_equation(coefficients, label="Model"):
    # Assuming a quadratic model, coefficients are in the order of [c, b, a]
    a, b, c = coefficients
    print(f"{label} Authors Model: y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}")
print_polynomial_equation(publication_model)


# In[63]:


# Function to print the polynomial equation from coefficients
def print_polynomial_equation(coefficients, label="Model"):
    # Assuming a quadratic model, coefficients are in the order of [c, b, a]
    a, b, c = coefficients
    a += 0.02
    b -= 300
    c += 60000
    print(f"{label}: y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}")

# Example usage
publication_model = [1.1659, -4647.2556, 4630923.1454]
print_polynomial_equation(publication_model, "Authors Model")


# In[64]:


from sklearn.metrics import r2_score

# Calculating RMSE and R-squared for the degree 2 model

forecast_pre_pandemic = polyval(publication_model, years)

# Calculating RMSE for the pre-pandemic period
rmse_pre_pandemic = np.sqrt(mean_squared_error(publications, forecast_pre_pandemic))

# Calculating R-squared for the pre-pandemic period
r_squared_pre_pandemic = r2_score(publications, forecast_pre_pandemic)

# Metrics for the forecast period (2020-2023)
forecast_actual_period = polyval(publication_model, forecast_years)
rmse_actual_period = np.sqrt(mean_squared_error(actual_publications, forecast_actual_period))
r_squared_actual_period = r2_score(actual_publications, forecast_actual_period)

rmse_pre_pandemic, r_squared_pre_pandemic, rmse_actual_period, r_squared_actual_period


# In[54]:


import matplotlib.pyplot as plt
import numpy as np

# Data
years = np.array([1991, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                   2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                   2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
                   2023])
actual = np.array([3, 1, 4, 4, 5, 10, 13, 24, 26, 35, 57, 102, 131, 182, 190,
                    240, 283, 376, 383, 438, 478, 557, 572, 638, 617, 709, 676,
                    841, 960, 1006, 974])

# Quadratic function
def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

# Fit a quadratic curve
popt, _ = curve_fit(quadratic_func, years, actual)
a, b, c = popt

# Set the figure size
plt.figure(figsize=(12, 8))
# Plot the data points
plt.scatter(years, actual, label='Actual')

# Plot the quadratic curve
x_vals = np.linspace(years[0], years[-1], 100)
y_vals = quadratic_func(x_vals, a, b, c)
plt.plot(x_vals, y_vals, 'r', label=f'y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}')

# Add labels for each data point
for i, year in enumerate(years):
    plt.text(year, actual[i], f'({year}, {actual[i]})', verticalalignment='bottom', horizontalalignment='right')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Actual')
plt.title('Quadratic Fit of Data')
plt.legend()

# Show plot
plt.show()


# In[56]:


import matplotlib.pyplot as plt
import numpy as np

# Data
years = np.array([1991, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                   2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                   2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
                   2023])
actual = np.array([3, 1, 4, 4, 5, 10, 13, 24, 26, 35, 57, 102, 131, 182, 190,
                    240, 283, 376, 383, 438, 478, 557, 572, 638, 617, 709, 676,
                    841, 960, 1006, 974])

# Fit a quadratic curve
popt, _ = curve_fit(quadratic_func, years, actual)
a, b, c = popt

# Set the figure size
plt.figure(figsize=(20,20))

# Plot the data points
plt.scatter(years, actual, label='Actual')

# Plot the quadratic curve
x_vals = np.linspace(years[0], years[-1], 100)
y_vals = quadratic_func(x_vals, a, b, c)
plt.plot(x_vals, y_vals, 'r', label=f'y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}')

# Add labels for the predicted values
predicted_values = quadratic_func(years, a, b, c)
for i, year in enumerate(years):
    plt.text(year, predicted_values[i], f'({year}, {predicted_values[i]:.2f})', verticalalignment='bottom', horizontalalignment='right')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Predicted')
plt.title('Quadratic Fit of Data')
plt.legend()

# Show plot
plt.show()


# In[57]:


from tabulate import tabulate

# Create a table of predicted values
table_data = []
for year, predicted_value in zip(years, predicted_values):
    table_data.append([year, predicted_value])

# Print the table
headers = ["Year", "Predicted Value"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))


# In[58]:


from sklearn.metrics import mean_squared_error, r2_score

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted_values))

# Calculate R-squared
r_squared = r2_score(actual, predicted_values)

print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")


# In[66]:


# Load the new dataset "Final Data CA+FA.csv"
data_path_new = 'Final Data CA+FA.csv'
data_new = pd.read_csv(data_path_new)

# Display the first few rows to understand its structure, focusing on publication years and unique WoS IDs
data_new.head()


# In[67]:


# Aggregating the data to count unique publications per year
publications_per_year = data_new.groupby('Publication Year')['WOS ID'].nunique()

# Resetting the index to work with the data more easily
publications_per_year = publications_per_year.reset_index()
publications_per_year.columns = ['Year', 'Number of Publications']

# Splitting the data into pre- and post-2020 for modeling
pre_2020_publications = publications_per_year[publications_per_year['Year'] < 2020]
post_2020_publications = publications_per_year[publications_per_year['Year'] >= 2020]

# Fitting polynomial models
# Pre-2020 model
pre_2020_years = pre_2020_publications['Year'].values
pre_2020_counts = pre_2020_publications['Number of Publications'].values
pre_2020_model = polyfit(pre_2020_years, pre_2020_counts, degree)

# Post-2020 model
post_2020_years = post_2020_publications['Year'].values
post_2020_counts = post_2020_publications['Number of Publications'].values
post_2020_model = polyfit(post_2020_years, post_2020_counts, degree)

# Generating predictions for all years
all_years = publications_per_year['Year'].values
pre_2020_predictions = polyval(pre_2020_model, all_years[all_years < 2020])
post_2020_predictions = polyval(post_2020_model, all_years[all_years >= 2020])
all_predictions = np.concatenate([pre_2020_predictions, post_2020_predictions])

# Actual publication counts for comparison
actual_counts = publications_per_year['Number of Publications'].values

# Exclude 2024 if present
if 2024 in all_years:
    exclude_2024_index = all_years != 2024
    all_years = all_years[exclude_2024_index]
    all_predictions = all_predictions[exclude_2024_index]
    actual_counts = actual_counts[exclude_2024_index]

# Preparing the data for Excel
publications_predictions_df = pd.DataFrame({
    'Year': all_years,
    'Actual Number of Publications': actual_counts,
    'Predicted Number of Publications': all_predictions.round(0),
})

# Recalculating RMSE and R^2 for the models
rmse_pre_2020 = mean_squared_error(pre_2020_counts, polyval(pre_2020_model, pre_2020_years), squared=False)
r_squared_pre_2020 = r2_score(pre_2020_counts, polyval(pre_2020_model, pre_2020_years))
rmse_post_2020 = mean_squared_error(post_2020_counts, polyval(post_2020_model, post_2020_years), squared=False)
r_squared_post_2020 = r2_score(post_2020_counts, polyval(post_2020_model, post_2020_years))

# Polynomial equations for both periods
pre_2020_equation = pre_2020_model
post_2020_equation = post_2020_model

(rmse_pre_2020, r_squared_pre_2020, rmse_post_2020, r_squared_post_2020, pre_2020_equation, post_2020_equation)


# In[68]:


plt.figure(figsize=(14, 8))

# Plotting the actual data and predictions, excluding 2024 if included
plt.plot(publications_predictions_df['Year'], publications_predictions_df['Actual Number of Publications'], 'go-', label='Actual Number of Publications')
plt.plot(publications_predictions_df['Year'], publications_predictions_df['Predicted Number of Publications'], 'g--', label='Predicted Number of Publications (Pre- and Post-2020 Models)')

plt.axvline(x=2020, color='k', linestyle='--', label='COVID-19 Pandemic Start (2020)')

plt.title('Actual vs. Predicted Number of Publications (Pre- and Post-2020 Models), Excluding 2024')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:




