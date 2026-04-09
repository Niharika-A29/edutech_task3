import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the dataset
df = sns.load_dataset('iris')
print("Dataset Loaded")
print("Shape: ", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nSpecies types:", df['species'].unique())

print("\ndataset info:")
df.info()

print("\nstatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

df.hist(figsize=(10, 8), bins=20, color='steelblue', edgecolor='black') 
plt.suptitle("Histogram of All numerical columns", y=1.02)
plt.tight_layout()
plt.savefig('histograms.png')
print("histograms Saved!")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, col in enumerate(columns):
    row = i // 2
    col_idx = i % 2
    sns.boxplot(y=df[col], ax=axes[row][col_idx], color='lightblue')
    axes[row][col_idx].set_title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('boxplots_iris.png')
print("Boxplots Saved!")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='Set1')
plt.title("petal length vs petal width by species")
plt.savefig('scatter_petal.png')
print("scatter plot Saved!")

plt.figure(figsize=(8, 6))
species_means = df.groupby('species').mean()
species_means.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title("average measurements by species")
plt.xlabel("Species")
plt.ylabel("measurement (cm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("bar_chart.png")
print("Bar chart Saved!")

plt.figure(figsize=(8, 6))
correlation = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
print("\nCorrelation matrix:")
print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm',  fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap Saved!")

pairplot = sns.pairplot(df, hue='species', palette='Set1')
pairplot.savefig('pairplot.png')
print("Pairplot Saved!")

print("\n" + "="*50)
print("Key Insights from eda")
print("="*50)
print("\n1. Dataset Shape:", df.shape)
print("2. Species:", df['species'].unique())
print("3. Missing values: None - clean dataset")
print("\n4. correlation findings:")
print(correlation.to_string())
print("\n5. Average petal length per species:")
print(df.groupby('species')['petal_length'].mean())
print("\n6. Average petal width per species:")
print(df.groupby('species')['petal_width'].mean())