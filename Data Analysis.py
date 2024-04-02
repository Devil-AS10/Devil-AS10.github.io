#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the data
file_path = 'Final Data CA+FA.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()


# In[2]:


# Split the data based on the 'Author Type'
first_authors = data[data['Author Type'] == 'First']
corresponding_authors = data[data['Author Type'] == 'Corresponding']

# Count the number of authors by country for each group
first_authors_count = first_authors['Country'].value_counts().head(10)
corresponding_authors_count = corresponding_authors['Country'].value_counts().head(10)

first_authors_count, corresponding_authors_count


# In[7]:


# Filter the data for male and female authors separately for both first and corresponding authors
male_first_authors = first_authors[first_authors['Gender'] == 'male']['Country'].value_counts().head(10)
female_first_authors = first_authors[first_authors['Gender'] == 'female']['Country'].value_counts().head(10)

male_corresponding_authors = corresponding_authors[corresponding_authors['Gender'] == 'male']['Country'].value_counts().head(10)
female_corresponding_authors = corresponding_authors[corresponding_authors['Gender'] == 'female']['Country'].value_counts().head(10)

# Create a DataFrame for easier plotting
top_countries = set(male_first_authors.index.tolist() + female_first_authors.index.tolist() +
                    male_corresponding_authors.index.tolist() + female_corresponding_authors.index.tolist())

top_countries_list = sorted(list(top_countries))

# Reinitialize the DataFrame with the sorted list of top countries
country_gender_counts = pd.DataFrame(index=top_countries_list, columns=['Male First Authors', 'Female First Authors',
                                                                        'Male Corresponding Authors', 'Female Corresponding Authors'])

# Fill the DataFrame with the counts for a more accurate plotting
for country in top_countries_list:
    country_gender_counts.loc[country, 'Male First Authors'] = male_first_authors.get(country, 0)
    country_gender_counts.loc[country, 'Female First Authors'] = female_first_authors.get(country, 0)
    country_gender_counts.loc[country, 'Male Corresponding Authors'] = male_corresponding_authors.get(country, 0)
    country_gender_counts.loc[country, 'Female Corresponding Authors'] = female_corresponding_authors.get(country, 0)

country_gender_counts.fillna(0, inplace=True)  # Ensure there are no NaN values
country_gender_counts = country_gender_counts.astype(int)  # Convert counts to integers for plotting

country_gender_counts


# In[12]:


# Recalculate counts for male and female authors for both First and Corresponding Authors for all countries
male_fa_counts = first_authors[first_authors['Gender'] == 'male'].groupby('Country').size()
female_fa_counts = first_authors[first_authors['Gender'] == 'female'].groupby('Country').size()

male_ca_counts = corresponding_authors[corresponding_authors['Gender'] == 'male'].groupby('Country').size()
female_ca_counts = corresponding_authors[corresponding_authors['Gender'] == 'female'].groupby('Country').size()

# Combine these counts into a new DataFrame for easier plotting and analysis
combined_counts = pd.DataFrame({
    'Male FA': male_fa_counts,
    'Female FA': female_fa_counts,
    'Male CA': male_ca_counts,
    'Female CA': female_ca_counts
}).fillna(0).astype(int)  # Fill missing values with 0 and ensure counts are integers

# Sort this combined data for First Authors and Corresponding Authors separately in descending order
sorted_fa_combined = combined_counts[['Male FA', 'Female FA']].sum(axis=1).sort_values(ascending=False).head(10)
sorted_ca_combined = combined_counts[['Male CA', 'Female CA']].sum(axis=1).sort_values(ascending=False).head(10)

# Now retrieve the detailed counts for top countries for FA and CA separately
top_fa_countries = combined_counts.loc[sorted_fa_combined.index]
top_ca_countries = combined_counts.loc[sorted_ca_combined.index]

top_fa_countries, top_ca_countries


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
# Adjusted function to plot side-by-side bars without internal value labels, only on top
def plot_author_type_counts_side_by_side_adjusted(data, title, columns, ax):
    sorted_data = data[columns].sum(axis=1).sort_values(ascending=False)
    data_sorted = data.loc[sorted_data.index]

    ind = np.arange(len(data_sorted))  # the x locations for the groups

    # Plotting the bars side by side with adjusted colors for colorblind-friendly visualization
    ax.bar(ind - width/2, data_sorted[columns[0]], width, color='tab:blue', label='Male')
    ax.bar(ind + width/2, data_sorted[columns[1]], width, color='tab:orange', label='Female')

    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(data_sorted.index, rotation='vertical')
    ax.legend()

# Re-create figure and axes for the subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

# Plot for First Authors with labels only on top
plot_author_type_counts_side_by_side_adjusted(top_fa_countries, 'Top Countries for First Authors (FA)', ['Male FA', 'Female FA'], ax1)
add_value_labels_on_top(ax1)

# Plot for Corresponding Authors with labels only on top
plot_author_type_counts_side_by_side_adjusted(top_ca_countries, 'Top Countries for Corresponding Authors (CA)', ['Male CA', 'Female CA'], ax2)
add_value_labels_on_top(ax2)

plt.tight_layout()
plt.show()



# In[18]:


# Load the new data
new_file_path = 'High Confidence Data - Sheet2.csv'
new_data = pd.read_csv(new_file_path)

# Display the first few rows of the dataframe to understand its structure
new_data.head()


# In[21]:


# Split the new data based on the 'Author Type' to differentiate between first authors and corresponding authors
new_first_authors = new_data[new_data['Author Type'] == 'First']
new_corresponding_authors = new_data[new_data['Author Type'] == 'Corresponding']

new_male_fa_counts = new_first_authors[new_first_authors['Gender'] == 'male'].groupby('Country').size()
new_female_fa_counts = new_first_authors[new_first_authors['Gender'] == 'female'].groupby('Country').size()

new_male_ca_counts = new_corresponding_authors[new_corresponding_authors['Gender'] == 'male'].groupby('Country').size()
new_female_ca_counts = new_corresponding_authors[new_corresponding_authors['Gender'] == 'female'].groupby('Country').size()


# Combine these counts into a new DataFrame for easier plotting and analysis in the new dataset
new_combined_counts = pd.DataFrame({
    'Male FA': new_male_fa_counts,
    'Female FA': new_female_fa_counts,
    'Male CA': new_male_ca_counts,
    'Female CA': new_female_ca_counts
}).fillna(0).astype(int)  # Fill missing values with 0 and ensure counts are integers

new_combined_counts.head()


# In[22]:


# Sort the combined data for First Authors and Corresponding Authors separately in descending order based on total counts
sorted_new_fa_combined = new_combined_counts[['Male FA', 'Female FA']].sum(axis=1).sort_values(ascending=False).head(10)
sorted_new_ca_combined = new_combined_counts[['Male CA', 'Female CA']].sum(axis=1).sort_values(ascending=False).head(10)

# Retrieve the detailed counts for top countries for FA and CA separately
top_new_fa_countries = new_combined_counts.loc[sorted_new_fa_combined.index]
top_new_ca_countries = new_combined_counts.loc[sorted_new_ca_combined.index]

# Create separate, side-by-side bar graphs for First Authors and Corresponding Authors, including value labels
# Function to plot side-by-side bars for the new data
def plot_new_author_type_counts_side_by_side(data, title, columns, ax):
    sorted_data = data[columns].sum(axis=1).sort_values(ascending=False)
    data_sorted = data.loc[sorted_data.index]

    ind = np.arange(len(data_sorted))  # the x locations for the groups

    # Plotting the bars side by side
    ax.bar(ind - width/2, data_sorted[columns[0]], width, color='tab:blue', label='Male')
    ax.bar(ind + width/2, data_sorted[columns[1]], width, color='tab:orange', label='Female')

    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(data_sorted.index, rotation='vertical')
    ax.legend()

# Re-create figure and axes for the subplots for the new data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

# Plot for First Authors with labels only on top
plot_new_author_type_counts_side_by_side(top_new_fa_countries, 'Top Countries for First Authors (FA)', ['Male FA', 'Female FA'], ax1)
add_value_labels_on_top(ax1)

# Plot for Corresponding Authors with labels only on top
plot_new_author_type_counts_side_by_side(top_new_ca_countries, 'Top Countries for Corresponding Authors (CA)', ['Male CA', 'Female CA'], ax2)
add_value_labels_on_top(ax2)

plt.tight_layout()
plt.show()


# In[ ]:




