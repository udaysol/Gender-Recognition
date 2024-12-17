import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the extracted features
data = pd.read_csv('extracted_features_with_labels.csv')

# Filter out additional labels, keeping only male and female
data = data[data['gender'].isin(['male_masculine', 'female_feminine'])]

# Separate features and labels
X = data.drop(columns=['gender', 'file_path'])
y = data['gender']

# Encode labels (0 for male, 1 for female)
le = LabelEncoder()
y = le.fit_transform(y)
print(f"Classes in LabelEncoder: {le.classes_}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test to pandas Series to save them properly
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)
print(y_train_series.value_counts())


# Save split data
X_train.to_csv('x_train.csv', index=False)
X_test.to_csv('x_test.csv', index=False)
y_train_series.to_csv('y_train.csv', index=False)
y_test_series.to_csv('y_test.csv', index=False)

print("Data preparation completed. Files saved as x_train.csv, x_test.csv, y_train.csv, and y_test.csv.")
