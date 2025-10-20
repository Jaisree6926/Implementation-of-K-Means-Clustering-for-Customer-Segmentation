# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
* Choose the number of clusters (k) — Decide how many groups you want to form.
* Initialize centroids — Randomly place k centroids in the dataset.
* Assign points to clusters — Each data point is assigned to the nearest centroid.
* Update centroids — Calculate new centroids as the mean of all points in each cluster.
* Repeat steps 3–4 until centroids stop changing (convergence).

## Program:
```
/*
      Program to implement the K Means Clustering for Customer Segmentation.
      Developed by: Jaisree B
      RegisterNumber:  212224230100
      */
      
      # Importing libraries
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from sklearn.cluster import KMeans
      
      # Importing the dataset
      data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\sem 3\ml\Mall_Customers.csv")
      X = data.iloc[:, [3, 4]].values   # Selecting Annual Income and Spending Score columns
      
      # Using the Elbow Method to find the optimal number of clusters
      wcss = []
      for i in range(1, 11):
          kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
          kmeans.fit(X)
          wcss.append(kmeans.inertia_)
      
      plt.plot(range(1, 11), wcss)
      plt.title('The Elbow Method')
      plt.xlabel('Number of clusters')
      plt.ylabel('WCSS')
      plt.show()
      
      # Applying K-Means with the optimal number of clusters (e.g., 5)
      kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
      y_kmeans = kmeans.fit_predict(X)
      
      # Visualizing the clusters
      plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
      plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
      plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
      plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
      plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
      
      # Plotting centroids
      plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                  s=300, c='yellow', label='Centroids')
      
      plt.title('Customer Segments')
      plt.xlabel('Annual Income (k$)')
      plt.ylabel('Spending Score (1–100)')
      plt.legend()
      plt.show()
```

## Output:
<img width="859" height="559" alt="Screenshot 2025-10-20 133107" src="https://github.com/user-attachments/assets/2da93884-8cef-4b26-aa4c-8cc7b099cc13" />

<img width="930" height="552" alt="Screenshot 2025-10-20 133100" src="https://github.com/user-attachments/assets/5cc11dd5-6d38-42b0-b42e-31e05b63ecb6" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
