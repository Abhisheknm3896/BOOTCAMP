from openpyxl.styles import named_styles
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding dataset")

file_name='sales_data.csv'
if not os.path.exists(file_name):
    print(f"error:{file_name}is not found")
    exit()

df=pd.read_csv(file_name)
print("Successfully loaded ")
print(f"Shape of the dataset:Rows:{df.shape[0]},Columns:{df.shape[1]}")

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

print("Handling Missing values")

print(df.isnull().sum())

median_age=df['Age'].median()
df['Age']=df['Age'].fillna(median_age)
print(median_age)

mean_spending = df['Spending'].mean()
df['Spending']=df['Spending'].fillna(mean_spending)
print(mean_spending)

plt.figure(figsize=(10,6))
df['Spending'].hist(bins=20,edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Spending')
plt.ylabel('Frequency')


correlation=df.corr(numeric_only=True)
print(correlation)
plt.show()

print("ploting correlation Heatmap")
plt.figure(figsize=(10,8))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

print("Find  the Outies in age")
outliers=df[df['Age']>100]
print("Found Outlires(s):")
print(outliers)