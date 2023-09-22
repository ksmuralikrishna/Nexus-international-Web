import pandas as pd

# Read the CSV file
df = pd.read_csv('symptom_precaution.csv')

# Combine the columns into a new column
df['Combined_Precautions'] = df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# Save the modified DataFrame back to the CSV file
df.to_csv('symptom_precaution1.csv', index=False)
