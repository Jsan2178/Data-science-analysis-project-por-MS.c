# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.offline as py
import plotly.figure_factory as ff
%matplotlib inline
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
dataset.head()
dataset.describe()

print("Mean of Annual Income (k$) of Female:",dataset['Annual Income (k$)'].loc[dataset['Genre'] == 'Female'].mean())
print("Mean of Annual Income (k$) of Male:",dataset['Annual Income (k$)'].loc[dataset['Genre'] == 'Male'].mean())

corr= dataset.select_dtypes(include="number").corr() #just include numerical features not object
plt.figure(figsize=(10,5))
sns.heatmap(corr,annot=True,cmap='jet',fmt='.4f',linewidths=1)
plt.show()

#Grouping by Gender
#let's groupby gender
dataset.groupby("Genre").mean()

init_notebook_mode(connected=True)
grouped = dataset["Genre"].value_counts(sort=False).rename_axis("Genre").reset_index(name="count")
grouped = grouped.rename(columns = {"index" : col, 0: "count"})

## let's plot this
colors = ['magenta', 'turquoise']
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0],marker=dict(colors=colors, line=dict(color='#000000', width=2)))
layout = {'title': 'Gender(Male, Female)'}
fig = go.Figure(data = [trace], layout = layout)
fig.update_layout(width=700, height=500,            #bigger
                  margin=dict(l=10, r=10, t=60, b=10))

fig.update_traces(domain=dict(x=[0.08, 0.92],       # take sup more canvas
                              y=[0.10, 0.92]),
                  textposition='inside')

fig.update_layout(legend=dict(orientation='h',      # moves legend up
                              y=1.005, x=0.5, xanchor='center'))
iplot(fig)


### As we saw above, we can infer females are visiting malls more than males.

# Separate by gender
d1 = dataset[dataset['Genre'] == 'Male']
d2 = dataset[dataset['Genre'] == 'Female']
col = 'Age'  
# --- Counting ---
v1 = (d1[col].value_counts(sort=False)     #  -> count bye age 
          .rename_axis(col)                # the index (ages)
          .reset_index(name='count')       # pass DataFrame with columns ['Age','count']
          .sort_values(col))               # order by age

v2 = (d2[col].value_counts(sort=False)
          .rename_axis(col)
          .reset_index(name='count')
          .sort_values(col))


trace1 = go.Scatter(x=v1[col], y=v1['count'], name='Male',
                    marker=dict(color='#ff7f0e'))
trace2 = go.Scatter(x=v2[col], y=v2['count'], name='Female',
                    marker=dict(color='#a678de'))

layout = {'title':'Age count [ Male vs Female ]',
          'xaxis':{'title':'Age'},
          'yaxis':{'title':'Count'},
          'width': 900,        
          'height': 400,      
          'margin': {'l':40,'r':10,'t':60,'b':40}
}
fig = go.Figure(data=[trace1, trace2],layout=layout)
fig.layout.template='presentation'
iplot(fig)

#The above plot tells us that females between the age of 23 and 49 are visiting mall more frequently whereas male between the age of 19 and 48 frequent the mall more, but even so the females visit the mall more than males.

### Taking the values of Anual Income and Spending Score

X = dataset.iloc[:, [3, 4]].values
print(f"Data type and shape of X: {type(X)}, {X.shape}")
print(f"Values of X: \n{X}")

#for better view
dataset[["Annual Income (k$)","Spending Score (1-100)"]].head()

## Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(12,6))
dendrogram=sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram plot')
plt.show()

## Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering as AG
hc=AG(n_clusters = 5, linkage='ward') #assuming euclidean metric with ward
y_hc=hc.fit_predict(X)

## Visualising the Clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()


## Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init=10, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

### As we can see, the curve isn't steep enough after after 5 clusters so we can take this number
#We want to find groups which aren't unlabeled in the data

## Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
#Each cluster makes a match by components, we can see it as follows
print(f"Values of : {y_kmeans}, \nThey must have the same shape as X in the following way X(n,m), y_kmeans(n,), let's see if they match: \n{y_kmeans.shape}, {X.shape}\n")
#we compare for each value of X a cluster is assigned
print(X[y_kmeans == 0,0]) #for cluster 0 the following X.values are labeled in it

## Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.legend()
plt.show()

## Analyzing the results 
 #+ ### *Cluster 1:* The people here are the average income and spending score. They might not be the prime target but can be considered with other techniques to increase their their spending score.
 
 #+ ### *Cluster 2:* We see people here often to spending more and have high income. They must be convinced with the mall facilities as these people are the prime sources of profit.
 
 #+ ### *Cluster 3:* We can see this people have low income but actually higher spending score. Maybe they love the mall services therefore they often buy products. The mall might not be target them because of the income but still will not lose them.
 
 #+ ### **Cluster 4:** Here we have our prime target because these people have higher income but for some reason low spending score, maybe they are not satisfied with the mall's services and products. As these people have the potential for increase their spending score, the mall authorities should try to add new services and facilities in order to attract these people.
 
 #+ ### *Cluster 5:* These people have low income and low spending score, this is quite reasonable, they know how to spend according to their income. So we can be least interested in people of this cluster.

