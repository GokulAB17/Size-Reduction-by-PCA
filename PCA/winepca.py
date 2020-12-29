#Perform Principal component analysis and perform clustering using first 
#3 principal component scores (both heirarchial 
#and k mean clustering(scree plot or elbow curve) and obtain 
#optimum number of clusters and check whether 
#we have obtained same number of clusters with the original data 
#(class column we have ignored at the begining who shows it has 3 clusters)df
import pandas as pd 
import numpy as np

#loading wine dataset
w = pd.read_csv(r"filepath\wine.csv")

#EDA
w.describe()
w.head()
w.isnull().sum()


#importing PCA and other imp tools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
w_new = w.iloc[:,1:]
w_new.head(4)

# Normalizing the numerical data 
w_normal = pd.DataFrame(scale(w_new))


############## Hierchical clustering#################
from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch  

z = linkage(w_normal, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(w_normal) 


h=pd.Series(h_complete.labels_)

####################Kmeans clustering###################

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(w_normal)
k=pd.Series(kmeans.labels_)

#########################Applying PCA to dataset############################
#Performing PCA on dataset by considering no of components as 13 which is equal to no of predictors
pca = PCA(n_components = 13)
pca_values = pca.fit_transform(w_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 and PCA3 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:,3]
plt.scatter(x,y,z)


################### Clustering with PCA ##########################
new_df = pd.DataFrame(pca_values[:,0:9])
kmeans = KMeans(n_clusters=3)
kmeans.fit(new_df)
k_pca=pd.Series(kmeans.labels_)

############## Hierchical clustering#################
z = linkage(new_df, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(new_df) 
h_pca=pd.Series(h_complete.labels_)



#Arranging all clustering by hclust and kmeans  using PCA and not PCA in the main dataset for comparison
w_new["Hclust"]=h
w_new["Hclust_PCA"]=h_pca
w_new["Kclust"]=k
w_new["Kclust_PCA"]=k_pca
w_new=w_new.iloc[:,[13,14,15,16,0,1,2,3,4,5,6,7,8,9,10,11,12]]
w_new.head()

#Storing new data as csv file in system
w_new.to_csv(("winePCA.csv", encoding="utf-8")
         
# From clustering with PCA and without PCA we can see many similarities 
#From hclust and hclustpca numbering are in both but there are some mismatches
#From kclust and kclustpca numbering are diff 2 ~ 1, 0 ~ 0,1~2 but there are mismatches
#But it seems PCA is more aggravated towards performance than accuracy     
         