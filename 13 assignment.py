#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Visualize categorical variables using bar charts
categorical_variables = [ 'Application mode', 'Course',
                          'Previous qualification', 'Educational special needs',
                          'Scholarship holder', 'Target']

plt.figure(figsize=(15, 10))

sns.countplot(x=var, data=df, hue='Target', palette='viridis')
plt.title(f'Distribution of Target')

plt.tight_layout()
plt.show()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the data
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Visualize categorical variables using bar charts
categorical_variables = ['Application mode', 'Course', 'Previous qualification',
                          'Educational special needs', 'Scholarship holder', 'Target']

st.title('Distribution of Categorical Variables')

for var in categorical_variables:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=var, data=df, hue='Target', palette='viridis')
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot()

# Display a subset of the data using Streamlit
st.title('Subset of the Data')
st.dataframe(df.head())

