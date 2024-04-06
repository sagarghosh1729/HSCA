#importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
from scipy import linalg
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df=pd.read_csv(r"...location to the file...\wisc.csv")
data=df.iloc[:,0:9]


label_encoder=LabelEncoder()
df.iloc[:,9]=label_encoder.fit_transform(df.iloc[:,9])

m=data.shape[0]
n=data.shape[1]


k=df.iloc[:,9].nunique()


print (m, n, k)






        

data_c=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        data_c[i,j]=df.iloc[i,j]



Y=np.zeros((m,n))
Y=data_c

data_m=np.zeros((m,n))



        

for i in range(m):
    s=np.linalg.norm(data_c[i,:])+0.02
    for j in range(n):
        data_m[i,j]=data_c[i,j]/s



true_labels=np.zeros(m)
for i in range(m):
    true_labels[i]=df.iloc[i,9]

    

X=data_m




# Fast Euclidean Spectral Clustering
m_0=100

kmeans=KMeans(n_clusters=m_0)
kmeans.fit(data_c)
labels=kmeans.predict(data_c)
centroids=kmeans.cluster_centers_

Y=centroids
def euclidean_dist(X,Y):
    d=np.linalg.norm(X-Y)
    return d


m_1=Y.shape[0]
E=np.zeros((m_1,m_1))
for i in range(m_1):
    for j in range(m_1):
        E[i][j]=euclidean_dist(Y[i,:],Y[j,:])


epsilon=500000
hyp_par=0.1
Aff=np.zeros((m_1,m_1))
for i in range(m_1):
    for j in range(m_1):
        if(E[i][j]<epsilon):
            Aff[i][j]=np.exp(-hyp_par*E[i][j]*E[i][j])
        else:
            Aff[i][j]=0




Aff_p=np.zeros((m_1,m_1))
for i in range(m_1):
    for j in range(m_1):
        if(E[i][j]<epsilon):
            Aff_p[i][j]=np.exp(-hyp_par*E[i][j])
        else:
            Aff_p[i][j]=0



D_bar=np.zeros((m_1,m_1))
for i in range(m_1):
    for j in range(m_1):
        if(i==j):
            D_bar[i][j]=np.sum(Aff[i,:])
        else:
            D_bar[i][j]=0


D_bar_p=np.zeros((m_1,m_1))
for i in range(m_1):
    for j in range(m_1):
        if(i==j):
            D_bar[i][j]=np.sum(Aff_p[i,:])
        else:
            D_bar[i][j]=0
            

n_clusters=k
#spectral_cluster=SpectralClustering(n_clusters=n_clusters,affinity='precomputed', random_state=42)
L_bar=np.subtract(D_bar,Aff)
eigenvalues,eigenvectors=np.linalg.eigh(L_bar)
sorted_indices=np.argsort(eigenvalues)
eigenvectors=eigenvectors[:,sorted_indices]
n_clusters=k

PQ=eigenvectors[:,:n_clusters]
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(PQ)
cluster_labels=kmeans.labels_


L_bar_p=np.subtract(D_bar_p,Aff_p)
eigenvalues,eigenvectors=np.linalg.eigh(L_bar_p)
sorted_indices=np.argsort(eigenvalues)
eigenvectors=eigenvectors[:,sorted_indices]
n_clusters=k

PQ=eigenvectors[:,:n_clusters]
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(PQ)
cluster_labels_p=kmeans.labels_


fast_labels=np.zeros(m)
for i in range(m):
    for j in range(m_0):
        if(labels[i]==j):
            fast_labels[i]=int(cluster_labels[j])





fast_labels_p=np.zeros(m)
for i in range(m):
    for j in range(m_0):
        if(labels[i]==j):
            fast_labels_p[i]=int(cluster_labels_p[j])


print("With gaussian Kernel\n")
ari=adjusted_rand_score(true_labels, fast_labels)
nmi=normalized_mutual_info_score(true_labels,fast_labels)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))


print("With poisson Kernel\n")
ari=adjusted_rand_score(true_labels, fast_labels_p)
nmi=normalized_mutual_info_score(true_labels,fast_labels_p)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))





#Fast Hyperbolic Spectral Clustering
def poincare_dist(X,Y):
    pq=np.linalg.norm(X-Y)
    p=np.linalg.norm(X)
    q=np.linalg.norm(Y)
    s=1+((2*(pq**2))/(1-(p**2))*(1-(q**2)))
    d=math.acosh(s)
    return d


def custom_pairwise_distances(X, Y):
    return pairwise_distances(X, Y, metric=poincare_dist)

m_0=100

kmeans=KMeans(n_clusters=m_0)
kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

Y=centroids

m_1=Y.shape[0]

R=np.zeros((m_1,m_1))
for i in range(m_1):
    for j in range(m_1):
        R[i][j]=poincare_dist(X[i,:],X[j,:])


W=np.zeros((m_1,m_1))
for i in range(m_1):
     for j in range(m_1):
          if(R[i][j]<epsilon):
               W[i][j]=np.exp(-hyp_par*R[i][j]*R[i][j])
          else:
               W[i][j]=0


W_p=np.zeros((m_1,m_1))
for i in range(m_1):
     for j in range(m_1):
          if(R[i][j]<epsilon):
               W_p[i][j]=np.exp(-2*hyp_par*R[i][j])
          else:
               W_p[i][j]=0



n_clusters=k
spectral_cluster=SpectralClustering(n_clusters=n_clusters,affinity='precomputed', random_state=42)
cluster_labels=spectral_cluster.fit_predict(W)


fast_labels=np.zeros(m)
for i in range(m):
    for j in range(m_0):
        if(labels[i]==j):
            fast_labels[i]=int(cluster_labels[j])


cluster_labels=spectral_cluster.fit_predict(W_p)


fast_labels_p=np.zeros(m)
for i in range(m):
    for j in range(m_0):
        if(labels[i]==j):
            fast_labels_p[i]=int(cluster_labels[j])


print("With gaussian Kernel\n")
ari=adjusted_rand_score(true_labels, fast_labels)
nmi=normalized_mutual_info_score(true_labels,fast_labels)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))


print("With poisson Kernel\n")
ari=adjusted_rand_score(true_labels, fast_labels_p)
nmi=normalized_mutual_info_score(true_labels,fast_labels_p)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))




