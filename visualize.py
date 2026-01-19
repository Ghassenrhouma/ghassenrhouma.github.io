import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv('StudentsPerformance.csv')

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')

# Convert categorical to numerical for modeling
df_model = df.copy()
df_model['gender'] = df_model['gender'].map({'female': 0, 'male': 1})
df_model['lunch'] = df_model['lunch'].map({'free/reduced': 0, 'standard': 1})
df_model['test_preparation_course'] = df_model['test_preparation_course'].map({'none': 0, 'completed': 1})

# One-hot encode race/ethnicity and parental level of education
df_model = pd.get_dummies(df_model, columns=['race_ethnicity', 'parental_level_of_education'], drop_first=True)

# Visualization 1: Complexity - Pairplot of scores colored by gender
plt.figure(figsize=(10, 8))
sns.pairplot(df[['math_score', 'reading_score', 'writing_score', 'gender']], hue='gender', diag_kind='kde')
plt.suptitle('Pairplot of Test Scores Colored by Gender', y=1.02)
plt.savefig('viz1_complexity.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Trend - Average scores by parental education
education_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
df['parental_level_of_education'] = pd.Categorical(df['parental_level_of_education'], categories=education_order, ordered=True)
avg_scores = df.groupby('parental_level_of_education')[['math_score', 'reading_score', 'writing_score']].mean().reset_index()

plt.figure(figsize=(12, 6))
avg_scores.set_index('parental_level_of_education').plot(kind='bar', figsize=(12, 6))
plt.title('Average Test Scores by Parental Level of Education')
plt.ylabel('Average Score')
plt.xlabel('Parental Level of Education')
plt.xticks(rotation=45)
plt.legend(title='Subject')
plt.tight_layout()
plt.savefig('viz2_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Model - Predict math score
X = df_model.drop(['math_score'], axis=1)
y = df_model['math_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Feature importance (coefficients)
features = X.columns
coefficients = model.coef_

plt.figure(figsize=(10, 6))
plt.barh(features, coefficients)
plt.title('Feature Importance for Predicting Math Score')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('viz3_model.png', dpi=300, bbox_inches='tight')
plt.close()

print(f'Model MSE: {mse}')
print('Plots saved as viz1_complexity.png, viz2_trend.png, viz3_model.png')