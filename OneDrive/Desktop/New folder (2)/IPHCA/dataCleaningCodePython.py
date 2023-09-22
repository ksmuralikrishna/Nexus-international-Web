import pandas as pd

# Read the data into a DataFrame
df = pd.read_csv('test.csv')

# Check for missing values
df.isnull().sum()

# Drop rows with missing values
df = df.dropna()

# Clean up the data
df['Disease'] = df['Disease'].str.replace(' ', '')
df['Diagnosis Test'] = df['Diagnosis Test'].str.replace(' ', '')

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_disease_diagnosis_test.csv', index=False)