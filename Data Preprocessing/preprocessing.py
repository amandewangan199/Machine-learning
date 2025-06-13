# Titanic Data Preprocessing Project
# Author: Aman Dewangan
# For: TechnoHacks Vocational Training - Task 1

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Step 2: Load the Dataset
print("🔹 Loading dataset...")
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
print("✅ Dataset loaded successfully!\n")

# Step 3: Initial Exploration
print("🔍 Dataset Info:")
print(df.info())

print("\n🧮 Dataset Description:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

# Step 4: Handle Missing Values

# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Drop rows with missing 'Fare' if any
df.dropna(subset=['Fare'], inplace=True)

# Verify missing values are handled
print("✅ Missing values handled:")
print(df.isnull().sum())

# Step 5: Encode Categorical Columns

# Initialize LabelEncoder
le = LabelEncoder()

# Encode 'Sex' column: male=1, female=0
df['Sex'] = le.fit_transform(df['Sex'])

# Encode 'Embarked' column
df['Embarked'] = le.fit_transform(df['Embarked'])

# Convert 'Pclass' to string then encode
df['Pclass'] = df['Pclass'].astype(str)
df['Pclass'] = le.fit_transform(df['Pclass'])

print("✅ Categorical columns encoded.")

# Step 6: Normalize Numeric Columns

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize 'Age' and 'Fare'
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("✅ Numerical columns normalized.")

# Step 7: Save the Cleaned Dataset

# Save to a new CSV file
df.to_csv('titanic_cleaned.csv', index=False)

print("✅ Cleaned dataset saved as 'titanic_cleaned.csv'")