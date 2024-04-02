#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load and prepare the datasets
data_path = 'Final Data CA+FA.csv'
high_confidence_path = 'High Confidence Data.csv'
data = pd.read_csv(data_path)
high_confidence_data = pd.read_csv(high_confidence_path)


# In[3]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


male_cas = data[(data['Gender'] == 'male') & (data['Author Type'] == 'Corresponding') & (data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()
high_conf_male_cas = high_confidence_data[(high_confidence_data['Gender'] == 'male') & (high_confidence_data['Author Type'] == 'Corresponding') & (high_confidence_data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()

train_male_cas = male_cas[male_cas['Publication Year'] < 2024]
train_high_conf_male_cas = high_conf_male_cas[high_conf_male_cas['Publication Year'] < 2023]

years_to_predict = np.arange(1991, 2024)

# Fitting and predicting
def predict_values(train_data):
    poly_model = np.poly1d(np.polyfit(train_data['Publication Year'], train_data['WOS ID'], 2))
    ols_model = sm.OLS(train_data['WOS ID'], sm.add_constant(train_data['Publication Year'])).fit()
    poly_predictions = poly_model(years_to_predict)
    ols_predictions = ols_model.predict(sm.add_constant(years_to_predict))
    return poly_predictions, ols_predictions

poly_predictions_male_cas, ols_predictions_male_cas = predict_values(train_male_cas)
poly_predictions_high_conf, ols_predictions_high_conf = predict_values(train_high_conf_male_cas)

# Correcting t-tests for predictions vs actual values for 2020 and 2021
def perform_ttests(predicted_poly_values, predicted_ols_values, actual_values):
    actual_2020_2021 = actual_values[actual_values['Publication Year'].isin([2020, 2021])]['WOS ID'].values
    _, pvalue_poly = ttest_ind(predicted_poly_values[years_to_predict == 2020].tolist() + predicted_poly_values[years_to_predict == 2021].tolist(), actual_2020_2021)
    _, pvalue_ols = ttest_ind(predicted_ols_values[years_to_predict == 2020].tolist() + predicted_ols_values[years_to_predict == 2021].tolist(), actual_2020_2021)
    return pvalue_poly, pvalue_ols

pvalue_poly_male_cas, pvalue_ols_male_cas = perform_ttests(poly_predictions_male_cas, ols_predictions_male_cas, male_cas)
pvalue_poly_high_conf, pvalue_ols_high_conf = perform_ttests(poly_predictions_high_conf, ols_predictions_high_conf, high_conf_male_cas)

# Adjusting label positions dynamically and adding labels for predictions and actual values
def adjust_label_position_and_add_data(ax, year, predictions_poly, predictions_ols, actual_data):
    # Find actual value for the year if available
    actual_value = actual_data.loc[actual_data['Publication Year'] == year, 'WOS ID'].values[0] if year in actual_data['Publication Year'].values else None
    # Predicted values
    poly_value = predictions_poly[years_to_predict == year][0]
    ols_value = predictions_ols[years_to_predict == year][0]

    # Place labels
    ax.annotate(f'Actual: {actual_value}', (year, actual_value), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(f'Poly: {int(poly_value)}', (year, poly_value), textcoords="offset points", xytext=(0,-15), ha='center', color='green')
    ax.annotate(f'OLS: {int(ols_value)}', (year, ols_value), textcoords="offset points", xytext=(0,-30), ha='center', color='blue')

# Updating the function to print actual numbers for each category and focusing on all years
def plot_with_all_fixes(years, poly_predictions, ols_predictions, actual_data, title, pvalue_poly, pvalue_ols):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, poly_predictions, 'g--', label='Poly Predictions')
    ax.plot(years, ols_predictions, 'b--', label='OLS Predictions')
    ax.scatter(actual_data['Publication Year'], actual_data['WOS ID'], color='red', label='Actual')
    
    # Print header
    print(f"{'Year':>4} | {'Actual':>6} | {'Poly':>6} | {'OLS':>6}")
    print("-" * 30)
    
    # Iterating through all years to print actual vs predicted values
    for year in years:
        actual_value = actual_data[actual_data['Publication Year'] == year]['WOS ID'].sum() if year in actual_data['Publication Year'].values else 'N/A'
        poly_value = poly_predictions[years == year][0] if year in years else 'N/A'
        ols_value = ols_predictions[years == year][0] if year in years else 'N/A'
        print(f"{year:>4} | {actual_value:>6} | {int(poly_value):>6} | {int(ols_value):>6}")
    
    # Annotations only for years 2020 and 2021, as before
    for year in [2020, 2021]:
        adjust_label_position_and_add_data(ax, year, poly_predictions, ols_predictions, actual_data)
    
    ax.text(0.99, 0.01, f'Poly p-value: {pvalue_poly:.2e}\nOLS p-value: {pvalue_ols:.2e}', verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Male CAs')
    plt.show()

# You would call this updated function for each dataset as follows:
# plot_with_all_fixes_and_print(years_to_predict, poly_predictions_male_cas, ols_predictions_male_cas, male_cas, 'Overall Male CAs (Up to 2023)', pvalue_poly_male_cas, pvalue_ols_male_cas)
# plot_with_all_fixes_and_print(years_to_predict, poly_predictions_high_conf, ols_predictions_high_conf, high_conf_male_cas, 'High Confidence Male CAs (Up to 2023)', pvalue_poly_high_conf, pvalue_ols_high_conf)

# Overall dataset
plot_with_all_fixes(years_to_predict, poly_predictions_male_cas, ols_predictions_male_cas, male_cas, 'Overall Male CAs (Up to 2023)', pvalue_poly_male_cas, pvalue_ols_male_cas)

# High Confidence dataset
plot_with_all_fixes(years_to_predict, poly_predictions_high_conf, ols_predictions_high_conf, high_conf_male_cas, 'High Confidence Male CAs (Up to 2023)', pvalue_poly_high_conf, pvalue_ols_high_conf)


# In[4]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Assuming data is loaded as before

# Change to focus on female Corresponding Authors
female_cas = data[(data['Gender'] == 'female') & (data['Author Type'] == 'Corresponding') & (data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()
high_conf_female_cas = high_confidence_data[(high_confidence_data['Gender'] == 'female') & (high_confidence_data['Author Type'] == 'Corresponding') & (high_confidence_data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()

train_female_cas = female_cas[female_cas['Publication Year'] < 2024]
train_high_conf_female_cas = high_conf_female_cas[high_conf_female_cas['Publication Year'] < 2023]

# The rest of the functions remain the same, just apply them to the female datasets

poly_predictions_female_cas, ols_predictions_female_cas = predict_values(train_female_cas)
poly_predictions_high_conf_female, ols_predictions_high_conf_female = predict_values(train_high_conf_female_cas)

pvalue_poly_female_cas, pvalue_ols_female_cas = perform_ttests(poly_predictions_female_cas, ols_predictions_female_cas, female_cas)
pvalue_poly_high_conf_female, pvalue_ols_high_conf_female = perform_ttests(poly_predictions_high_conf_female, ols_predictions_high_conf_female, high_conf_female_cas)

# Use the plot_with_all_fixes function with female datasets
# For Overall Female CAs (Up to 2023)
plot_with_all_fixes(years_to_predict, poly_predictions_female_cas, ols_predictions_female_cas, female_cas, 'Overall Female CAs (Up to 2023)', pvalue_poly_female_cas, pvalue_ols_female_cas)

# For High Confidence Female CAs (Up to 2023)
plot_with_all_fixes(years_to_predict, poly_predictions_high_conf_female, ols_predictions_high_conf_female, high_conf_female_cas, 'High Confidence Female CAs (Up to 2023)', pvalue_poly_high_conf_female, pvalue_ols_high_conf_female)


# In[5]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Assuming data is loaded as before

# Change to focus on male First Authors
male_fas = data[(data['Gender'] == 'male') & (data['Author Type'] == 'First') & (data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()
high_conf_male_fas = high_confidence_data[(high_confidence_data['Gender'] == 'male') & (high_confidence_data['Author Type'] == 'First') & (high_confidence_data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()

train_male_fas = male_fas[male_fas['Publication Year'] < 2024]
train_high_conf_male_fas = high_conf_male_fas[high_conf_male_fas['Publication Year'] < 2023]

# The rest of the functions remain the same, just apply them to the male FA datasets

poly_predictions_male_fas, ols_predictions_male_fas = predict_values(train_male_fas)
poly_predictions_high_conf_male_fa, ols_predictions_high_conf_male_fa = predict_values(train_high_conf_male_fas)

pvalue_poly_male_fas, pvalue_ols_male_fas = perform_ttests(poly_predictions_male_fas, ols_predictions_male_fas, male_fas)
pvalue_poly_high_conf_male_fa, pvalue_ols_high_conf_male_fa = perform_ttests(poly_predictions_high_conf_male_fa, ols_predictions_high_conf_male_fa, high_conf_male_fas)

# Use the plot_with_all_fixes function with male FA datasets
# For Overall Male FAs (Up to 2023)
plot_with_all_fixes(years_to_predict, poly_predictions_male_fas, ols_predictions_male_fas, male_fas, 'Overall Male FAs (Up to 2023)', pvalue_poly_male_fas, pvalue_ols_male_fas)

# For High Confidence Male FAs (Up to 2023)
plot_with_all_fixes(years_to_predict, poly_predictions_high_conf_male_fa, ols_predictions_high_conf_male_fa, high_conf_male_fas, 'High Confidence Male FAs (Up to 2023)', pvalue_poly_high_conf_male_fa, pvalue_ols_high_conf_male_fa)


# In[6]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Assuming data is loaded as before

# Change to focus on female First Authors
female_fas = data[(data['Gender'] == 'female') & (data['Author Type'] == 'First') & (data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()
high_conf_female_fas = high_confidence_data[(high_confidence_data['Gender'] == 'female') & (high_confidence_data['Author Type'] == 'First') & (high_confidence_data['Publication Year'] < 2024)].groupby('Publication Year')['WOS ID'].size().reset_index()

train_female_fas = female_fas[female_fas['Publication Year'] < 2024]
train_high_conf_female_fas = high_conf_female_fas[high_conf_female_fas['Publication Year'] < 2023]

# The rest of the functions remain the same, just apply them to the female FA datasets

poly_predictions_female_fas, ols_predictions_female_fas = predict_values(train_female_fas)
poly_predictions_high_conf_female_fa, ols_predictions_high_conf_female_fa = predict_values(train_high_conf_female_fas)

pvalue_poly_female_fas, pvalue_ols_female_fas = perform_ttests(poly_predictions_female_fas, ols_predictions_female_fas, female_fas)
pvalue_poly_high_conf_female_fa, pvalue_ols_high_conf_female_fa = perform_ttests(poly_predictions_high_conf_female_fa, ols_predictions_high_conf_female_fa, high_conf_female_fas)

# Use the plot_with_all_fixes function with female FA datasets
# For Overall Female FAs (Up to 2023)
plot_with_all_fixes(years_to_predict, poly_predictions_female_fas, ols_predictions_female_fas, female_fas, 'Overall Female FAs (Up to 2023)', pvalue_poly_female_fas, pvalue_ols_female_fas)

# For High Confidence Female FAs (Up to 2023)
plot_with_all_fixes(years_to_predict, poly_predictions_high_conf_female_fa, ols_predictions_high_conf_female_fa, high_conf_female_fas, 'High Confidence Female FAs (Up to 2023)', pvalue_poly_high_conf_female_fa, pvalue_ols_high_conf_female_fa)


# In[9]:


# Assuming necessary libraries are already imported and datasets are loaded
# years_to_predict_final is defined as np.arange(1991, 2024)
min_year = 1991
max_year = 2023
years_to_predict_final = np.arange(min_year, max_year + 1)

# Helper function for predictions remains the same
def predict_values(train_data, years_to_predict):
    poly_model = np.poly1d(np.polyfit(train_data['Publication Year'], train_data['WOS ID'], 2))
    ols_model = sm.OLS(train_data['WOS ID'], sm.add_constant(train_data['Publication Year'])).fit()
    poly_predictions = poly_model(years_to_predict)
    ols_predictions = ols_model.predict(sm.add_constant(years_to_predict))
    return poly_predictions, ols_predictions

# High Confidence Data Predictions
# Male CAs
high_conf_male_cas = high_confidence_data[(high_confidence_data['Gender'] == 'male') & (high_confidence_data['Author Type'] == 'Corresponding')].groupby('Publication Year')['WOS ID'].size().reset_index()
train_high_conf_male_cas = high_conf_male_cas[high_conf_male_cas['Publication Year'] < 2020]
poly_predictions_high_conf_male_cas, ols_predictions_high_conf_male_cas = predict_values(train_high_conf_male_cas, years_to_predict_final)

# Female CAs
high_conf_female_cas = high_confidence_data[(high_confidence_data['Gender'] == 'female') & (high_confidence_data['Author Type'] == 'Corresponding')].groupby('Publication Year')['WOS ID'].size().reset_index()
train_high_conf_female_cas = high_conf_female_cas[high_conf_female_cas['Publication Year'] < 2020]
poly_predictions_high_conf_female_cas, ols_predictions_high_conf_female_cas = predict_values(train_high_conf_female_cas, years_to_predict_final)

# Male FAs
high_conf_male_fas = high_confidence_data[(high_confidence_data['Gender'] == 'male') & (high_confidence_data['Author Type'] == 'First')].groupby('Publication Year')['WOS ID'].size().reset_index()
train_high_conf_male_fas = high_conf_male_fas[high_conf_male_fas['Publication Year'] < 2020]
poly_predictions_high_conf_male_fas, ols_predictions_high_conf_male_fas = predict_values(train_high_conf_male_fas, years_to_predict_final)

# Female FAs
high_conf_female_fas = high_confidence_data[(high_confidence_data['Gender'] == 'female') & (high_confidence_data['Author Type'] == 'First')].groupby('Publication Year')['WOS ID'].size().reset_index()
train_high_conf_female_fas = high_conf_female_fas[high_conf_female_fas['Publication Year'] < 2020]
poly_predictions_high_conf_female_fas, ols_predictions_high_conf_female_fas = predict_values(train_high_conf_female_fas, years_to_predict_final)

# Printing High Confidence Data Predictions
print("High Confidence Data Predictions:")
print("Years:", years_to_predict_final)
print("Poly Predictions Male CA:", poly_predictions_high_conf_male_cas)
print("OLS Predictions Male CA:", ols_predictions_high_conf_male_cas)
print("Poly Predictions Female CA:", poly_predictions_high_conf_female_cas)
print("OLS Predictions Female CA:", ols_predictions_high_conf_female_cas)
print("Poly Predictions Male FA:", poly_predictions_high_conf_male_fas)
print("OLS Predictions Male FA:", ols_predictions_high_conf_male_fas)
print("Poly Predictions Female FA:", poly_predictions_high_conf_female_fas)
print("OLS Predictions Female FA:", ols_predictions_high_conf_female_fas)

# Repeat similar steps for Final Data (data) predictions and print them


# In[12]:


# Correcting the predict_values function to work with grouped data
def predict_values_corrected(train_data):
    # Using 'Publication Year' as x and 'Count' as y for fitting models
    poly_model = np.poly1d(np.polyfit(train_data['Publication Year'], train_data['Count'], 2))
    ols_model = sm.OLS(train_data['Count'], sm.add_constant(train_data['Publication Year'])).fit()
    poly_predictions = poly_model(years_to_predict)
    ols_predictions = ols_model.predict(sm.add_constant(years_to_predict))
    return poly_predictions, ols_predictions

# Adjusting the DataFrame creation to work with the fixed structure
def make_predictions_dataframe_corrected(category_data, years_to_predict, category_name):
    # Generating predictions
    poly_predictions, ols_predictions = predict_values_corrected(category_data)

    # Creating DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Year': years_to_predict,
        f'Poly Predictions {category_name}': poly_predictions,
        f'OLS Predictions {category_name}': ols_predictions
    })

    # Merging with actual counts data
    combined_df = predictions_df
    combined_df[f'Actual {category_name}'] = category_data.set_index('Publication Year').reindex(years_to_predict)['Count'].reset_index(drop=True).fillna(0).astype(int)

    return combined_df

# Re-applying the corrected function across categories
corrected_prediction_dfs = {}
for gender, author_type in categories:
    category_name = f"{gender}_{author_type}"
    category_data = data[(data['Gender'] == gender) & (data['Author Type'] == author_type)].groupby('Publication Year').size().reset_index(name='Count')
    
    # Ensuring the training data is before 2020 for predictions
    train_data = category_data[category_data['Publication Year'] < 2020]
    
    corrected_prediction_dfs[category_name] = make_predictions_dataframe_corrected(train_data, years_to_predict, category_name)

# Example output for Male Corresponding Authors after correction
corrected_prediction_dfs['male_Corresponding'].head()


# In[ ]:




