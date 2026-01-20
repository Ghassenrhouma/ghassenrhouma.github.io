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

# Visualization 1: Understanding Score Distributions and Relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Student Performance Analysis: Score Distributions and Relationships', fontsize=16, fontweight='bold')

# Histogram of math scores
axes[0,0].hist(df['math_score'], bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
axes[0,0].set_title('Mathematics Score Distribution')
axes[0,0].set_xlabel('Score')
axes[0,0].set_ylabel('Number of Students')
axes[0,0].grid(True, alpha=0.3)

# Histogram of reading scores
axes[0,1].hist(df['reading_score'], bins=20, alpha=0.7, color='#A23B72', edgecolor='black')
axes[0,1].set_title('Reading Score Distribution')
axes[0,1].set_xlabel('Score')
axes[0,1].set_ylabel('Number of Students')
axes[0,1].grid(True, alpha=0.3)

# Histogram of writing scores
axes[1,0].hist(df['writing_score'], bins=20, alpha=0.7, color='#F18F01', edgecolor='black')
axes[1,0].set_title('Writing Score Distribution')
axes[1,0].set_xlabel('Score')
axes[1,0].set_ylabel('Number of Students')
axes[1,0].grid(True, alpha=0.3)

# Correlation heatmap
corr = df[['math_score', 'reading_score', 'writing_score']].corr()
sns.heatmap(corr, annot=True, cmap='Blues', ax=axes[1,1], cbar_kws={'shrink': 0.8})
axes[1,1].set_title('Score Correlations')

plt.tight_layout()
plt.savefig('viz1_complexity.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 2: The Impact of Parental Education on Academic Performance
education_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
df['parental_level_of_education'] = pd.Categorical(df['parental_level_of_education'], categories=education_order, ordered=True)
avg_scores = df.groupby('parental_level_of_education', observed=False)[['math_score', 'reading_score', 'writing_score']].mean().reset_index()

plt.figure(figsize=(12, 7))
x = np.arange(len(avg_scores))
width = 0.25

plt.bar(x - width, avg_scores['math_score'], width, label='Mathematics', color='#2E86AB', alpha=0.8)
plt.bar(x, avg_scores['reading_score'], width, label='Reading', color='#A23B72', alpha=0.8)
plt.bar(x + width, avg_scores['writing_score'], width, label='Writing', color='#F18F01', alpha=0.8)

plt.xlabel('Parental Level of Education', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.title('Academic Performance by Parental Education Level', fontsize=14, fontweight='bold')
plt.xticks(x, [label.replace(' ', '\n') for label in avg_scores['parental_level_of_education']], rotation=0)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(60, 80)
plt.tight_layout()
plt.savefig('viz2_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 3: Key Factors Influencing Mathematics Achievement
# Calculate average scores by test preparation and lunch
prep_scores = df.groupby('test_preparation_course')['math_score'].mean()
lunch_scores = df.groupby('lunch')['math_score'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Test preparation impact
bars1 = ax1.bar(['No Preparation', 'Completed Course'], prep_scores.values, color=['#F18F01', '#2E86AB'], alpha=0.8)
ax1.set_title('Impact of Test Preparation Course', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Math Score')
ax1.set_ylim(60, 75)
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom')

# Lunch impact
bars2 = ax2.bar(['Free/Reduced', 'Standard'], lunch_scores.values, color=['#A23B72', '#2E86AB'], alpha=0.8)
ax2.set_title('Impact of Lunch Program', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Math Score')
ax2.set_ylim(55, 75)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom')

plt.suptitle('External Factors Affecting Mathematics Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('viz3_model.png', dpi=150, bbox_inches='tight')
plt.close()

print('Professional visualizations generated successfully!')