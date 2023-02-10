#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv('Mall_Customers.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.isnull().sum()


# In[11]:


data=dataset[['Annual Income (k$)','Spending Score (1-100)']]


# In[9]:


dataset.columns


# In[30]:


np.random.seed(0)
x = np.random.rand(100,2)


# In[32]:


wc_ss=[]
for i in range(1,11):
    kmeans_clu=KMeans(n_clusters=i,random_state=56)
    kmeans_clu.fit(x)
    wc_ss.append(kmeans_clu.inertia_)


# In[33]:


plt.figure(figsize=(10,5))
plt.plot(range(1,11), wc_ss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[35]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, random_state = 56)
y_kmeans = kmeans.fit_predict(x)


# In[38]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'pink', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'orange', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'green', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'violet', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




