## LIBS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

## READ DATA
df = pd.read_csv(r'Student Depression Dataset.csv')
df.head(10)
df.isna().sum()
df.isnull().sum()

# REMOVING ID
df = df.drop(['id'], axis=1)

## CHANGING CATERORICAL TO NUMERICAL (GENDER)
df.loc[df['Gender'] == 'Male', 'Gender'] = 0
df.loc[df['Gender'] == 'Female', 'Gender'] = 1
df.head(3)
df['City'].value_counts()
df['Profession'].value_counts()

# Visualizations
sns.set(style="whitegrid")

# 1. **Countplot for Gender**
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", data=df, palette="Set2")
plt.title("Gender Distribution", fontsize=16)
plt.show()

# 2. **Histogram for Age**
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color="blue")
plt.title("Age Distribution", fontsize=16)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 3. **Boxplot for CGPA**
plt.figure(figsize=(8, 5))
sns.boxplot(x="Gender", y="CGPA", data=df, palette="muted")
plt.title("CGPA Distribution by Gender", fontsize=16)
plt.show()

# 4. **Barplot for Study Satisfaction**
plt.figure(figsize=(8, 5))
sns.barplot(x="Study Satisfaction", y="CGPA", hue="Gender", data=df, palette="coolwarm")
plt.title("Study Satisfaction vs CGPA by Gender", fontsize=16)
plt.show()

# 6. **Pie Chart for Suicidal Thoughts**
suicidal_counts = df['Have you ever had suicidal thoughts ?'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(suicidal_counts, labels=suicidal_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
plt.title("Suicidal Thoughts Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Work Pressure", y="Financial Stress", hue="Gender", data=df, palette="dark")
plt.title("Work Pressure vs Financial Stress by Gender", fontsize=16)
plt.show()
df

## REMOVING CITIES WITH LESS THAN 400 STUDENTS
cities_to_remove = df['City'].value_counts()[df['City'].value_counts() < 400]
df = df[~df['City'].isin(cities_to_remove.index)]
df['City'].value_counts()
df = df.loc[df['Profession']=='Student']

df['Profession'].value_counts()

## REMOVING THE ORIGINAL CITY AND OLD DEGREE COLUMN
df = df.drop(['City', 'Degree'], axis=1)
df = df.drop(['Profession'], axis=1)
df = df.drop(['Work Pressure'], axis=1)
df = df.drop(['Job Satisfaction'], axis=1)
df.loc[df['Sleep Duration'] == 'Less than 5 hourse', 'Sleep Duration'] = 0
df.loc[df['Sleep Duration'] == '5-6 hourse', 'Sleep Duration'] = 1
df.loc[df['Sleep Duration'] == '7-8 hourse', 'Sleep Duration'] = 2
df.loc[df['Sleep Duration'] == 'More than 8 hourse', 'Sleep Duration'] = 3
df['Sleep Duration'].value_counts()

df.loc[df['Dietary Habits'] == 'Healthy', 'Dietary Habits'] = 0
df.loc[df['Dietary Habits'] == 'Unhealthy', 'Dietary Habits'] = 1
df.loc[df['Dietary Habits'] == 'Moderate', 'Dietary Habits'] = 3
df['Dietary Habits'].value_counts()
df = df.drop(['Sleep Duration'], axis=1)
df.head(3)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Have you ever had suicidal thoughts ?'] = label_encoder.fit_transform(df['Have you ever had suicidal thoughts ?'])
df['Family History of Mental Illness'] = label_encoder.fit_transform(df['Family History of Mental Illness'])
df.head(5)

scaler = StandardScaler()
df[['Age', 'CGPA']] = scaler.fit_transform(df[['Age', 'CGPA']])
df[['Academic Pressure','Study Satisfaction']] = scaler.fit_transform(df[['Academic Pressure','Study Satisfaction']])
df

# Define features (X) and target (y)
X = df.drop(columns=['Depression'])
y = df['Depression']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)






























