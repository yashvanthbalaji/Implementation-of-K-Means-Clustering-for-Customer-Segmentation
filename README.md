# Implementation-of-K-Means-Clustering-for-Customer-Segmentation
### NAME: BALAJI A
### REG.NO: 212223040023
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries

2. Load the dataset (Mall_Customers.csv)

3. Check dataset info,Check for missing values

4. Select features: Annual Income and Spending Score

5. Standardize the selected features

6. Apply Elbow method for k = 1 to 10 and Plot

7. Apply Silhouette Score for k = 2 to 10 and Plot 

8. Fit K-Means model

9. Predict cluster labels and add cluster labels to the dataset

10.Compute cluster centers

11. Visualize clusters using scatter plot

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: BALAJI A
RegisterNumber: 212223040023
```
```py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------
# 1. Load the dataset
# ---------------------------------------
df = pd.read_csv("Mall_Customers.csv")  
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
display(df.head())

# ---------------------------------------
# 2. Check info and missing values
# ---------------------------------------
print("\nDataset Info:")
display(df.info())
print("\nMissing Values:")
display(df.isnull().sum())

# ---------------------------------------
# 3. Select features for clustering
# Using Income & Spending Score
# ---------------------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

print("\nFeatures Used:", features)

# ---------------------------------------
# 4. Standardize the data
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 5. Elbow Method to choose k
# ---------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 6. Silhouette Scores
# ---------------------------------------
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), sil_scores, marker='o', color="orange")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 7. Apply KMeans (Choose k=5 by elbow)
# ---------------------------------------
k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# ---------------------------------------
# 8. Cluster Centers in original units
# ---------------------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers_original, columns=features)
centers_df["Cluster"] = range(k_final)

print("\nCluster Centers (Original Values):")
display(centers_df.round(2))

# ---------------------------------------
# 9. Visualization of Clusters
# ---------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=70
)

# Show cluster centers
plt.scatter(
    centers_df["Annual Income (k$)"],
    centers_df["Spending Score (1-100)"],
    s=250,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means (k=5)")
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
<img width="665" height="389" alt="image" src="https://github.com/user-attachments/assets/3b5346e5-09c1-430a-bbdd-5f10a58b74da" />

<img width="679" height="232" alt="image" src="https://github.com/user-attachments/assets/f43ff672-b897-4bcc-9c53-2f2994489b12" />
<img width="722" height="564" alt="image" src="https://github.com/user-attachments/assets/1f66a296-b514-4e3d-b99b-a9501cad1b55" />
<img width="757" height="530" alt="image" src="https://github.com/user-attachments/assets/c43ae0d5-c9cf-413e-a89a-f36516c6e665" />
<img width="738" height="477" alt="image" src="https://github.com/user-attachments/assets/53af065c-3728-40e2-93c2-f013bb261c4f" />
<img width="915" height="719" alt="image" src="https://github.com/user-attachments/assets/7ba862d8-4b45-4f75-a7ba-2ec990ac00c6" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
