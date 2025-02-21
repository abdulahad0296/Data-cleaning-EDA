# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load Titanic Dataset from Seaborn
df = sns.load_dataset('titanic')

# Display first 5 rows
print("Initial Data Preview:")
print(df.head())

# ----------------------------
# DATA CLEANING & PREPROCESSING
# ----------------------------

# Check for missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing 'age' with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill 'embarked' missing values with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop 'deck' column due to high number of missing values
df.drop(columns='deck', inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ----------------------------
# ENCODING CATEGORICAL DATA
# ----------------------------

# Label Encoding for 'sex' and 'embarked'
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

# ----------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------

# 1. Survival Count Plot
sns.countplot(x='survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 2. Survival Rate by Gender
sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Survival Rate')
plt.show()

# 3. Class-wise Survival Distribution
sns.countplot(x='pclass', hue='survived', data=df)
plt.title('Survival Distribution by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# 4. Correlation Heatmap (Fixed)
plt.figure(figsize=(10, 6))

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

print("\n✅ Analysis Complete!")
