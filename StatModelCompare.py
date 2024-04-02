#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the data
data_path = 'Final Data CA+FA.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataframe to understand its structure
data.head()


# In[3]:


# Count unique WOS IDs
unique_publications_count = data["WOS ID"].nunique()

unique_publications_count


# In[4]:


# Aggregate data by publication year, excluding 2024
publications_per_year = data[data["Publication Year"] < 2024].groupby("Publication Year")["WOS ID"].nunique()

# Convert the Series to a DataFrame for further processing
publications_per_year_df = publications_per_year.reset_index()
publications_per_year_df.columns = ['Year', 'Publications']

publications_per_year_df


# In[6]:


from statsmodels.api import OLS
import numpy as np
import statsmodels.api as sm

# Filter the data to include only pre-pandemic years for model training (up to 2018)
training_data = publications_per_year_df[publications_per_year_df['Year'] <= 2018]


# Define independent variable (Year) and dependent variable (Publications)
X_train = training_data['Year']
y_train = training_data['Publications']

# Adding a constant to the model (intercept)
X_train = sm.add_constant(X_train)

# Fit the OLS model
model = OLS(y_train, X_train).fit()

# Prepare the data for prediction (2019-2023)
predict_years = pd.DataFrame({'Year': [2019, 2020, 2021, 2022, 2023]})
predict_years = sm.add_constant(predict_years)

# Make predictions for 2019-2023
predictions = model.predict(predict_years)

# Displaying the total number of unique publications
print(f"Total number of unique publications: {unique_publications_count}")

# Displaying the predictions
print("Predictions for the number of publications (2019-2023):")
print(predictions)


# In[9]:


import matplotlib.pyplot as plt

# Actual number of publications per year (for all years including 2019-2023 for comparison)
actual_publications = publications_per_year_df.set_index('Year')

# Prediction years and values for plotting
prediction_years = [2019, 2020, 2021, 2022, 2023]
prediction_values = predictions.values

# Plotting
plt.figure(figsize=(15, 10))
plt.plot(actual_publications.index, actual_publications['Publications'], label='Actual Publications', marker='o')
plt.scatter(prediction_years, prediction_values, color='red', label='Predicted Publications')

# Enhancing the plot
plt.title('Actual and Predicted Publications Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xticks(range(1990, 2024))  # Adjusting x-axis to show every year for clarity
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# In[13]:


# Extending the predictions to cover all years in the dataset
all_years = pd.DataFrame({'Year': range(data["Publication Year"].min(), 2024)})
all_years_with_constant = sm.add_constant(all_years)

# Making predictions for all years
all_predictions = model.predict(all_years_with_constant)


# Enhancing the graph by adding a marker for each predicted value and labels for each mark

plt.figure(figsize=(14, 7))

# Plotting actual publications for years with available data
plt.plot(publications_per_year_df['Year'], publications_per_year_df['Publications'], label='Actual Publications', marker='o', color='blue')

# Plotting predicted publications for the full range with markers
plt.plot(all_years['Year'], all_predictions, label='Predicted Publications', linestyle='--', marker='x', color='red')

# Adding labels for each predicted value
for i, txt in enumerate(all_predictions):
    plt.annotate(f"{int(txt)}", (all_years['Year'][i], all_predictions[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Enhancing the plot
plt.title('Actual vs. Predicted Publications Per Year with Labels')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xticks(range(data["Publication Year"].min(), 2024, 2))  # Adjusting x-axis to show every second year for clarity
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# In[ ]:




