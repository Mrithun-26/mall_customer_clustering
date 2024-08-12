import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv("Mall_Customers.csv")
# df.head(5)
df.drop("CustomerID", inplace=True, axis=1)


def disvis(column):
    mean_ = df[column].mean()
    min_ = df[column].min()
    max_ = df[column].max()
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], bins=10, kde=True)
    plt.axvline(mean_, color='r', linestyle='--', label=f'Mean: {mean_:.2f}')
    plt.axvline(min_, color='g', linestyle='--', label=f'Min: {min_}')
    plt.axvline(max_, color='b', linestyle='--', label=f'Max: {max_}')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

x = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
for i in x:
    disvis(i)
df["Gender"].value_counts()

gender_counts = df["Gender"].value_counts().reset_index()

plt.figure(figsize=(10, 6))
plt.pie(gender_counts["count"],labels=gender_counts["Gender"], autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
# plt.show()

sns.pairplot(df,hue="Gender",palette='husl')
# plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Gender', data=df)
plt.title('Age vs. Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
# plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=df)
plt.title('Annual Income vs. Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
# plt.show()

labelencoder = LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = labelencoder.fit_transform(df[i])
df.head()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.show()

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_pca)

df['PCA1'] = df_pca[:, 0]
df['PCA2'] = df_pca[:, 1]
centroids = df.groupby('Cluster').mean()[['PCA1', 'PCA2']].values
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=100, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroid', marker='X')

plt.legend()
plt.title('Customer Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()