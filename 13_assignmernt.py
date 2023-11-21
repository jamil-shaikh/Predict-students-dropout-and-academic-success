st.title('Distribution of Categorical Variables')

# Plot bar charts for each categorical variable
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
