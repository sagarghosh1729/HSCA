import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Input data matrix X

df=pd.read_csv(r"...location to the file...\wisc.csv")
data=df.iloc[:,0:9]


label_encoder=LabelEncoder()
df.iloc[:,9]=label_encoder.fit_transform(df.iloc[:,9])

m=data.shape[0]
n=data.shape[1]


k=df.iloc[:,9].nunique()









        

data_c=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        data_c[i,j]=df.iloc[i,j]



Y=np.zeros((m,n))
Y=data_c
Y=Y.T

data_m=np.zeros((m,n))



        

for i in range(m):
    s=np.linalg.norm(data_c[i,:])+0.02
    for j in range(n):
        data_m[i,j]=data_c[i,j]/s



true_labels=np.zeros(m)
for i in range(m):
    true_labels[i]=df.iloc[i,9]

    

X=data_c
X=X.T



m,n=X.shape


p=50

kmeans=KMeans(n_clusters=p)
kmeans.fit(data_c)
labels=kmeans.predict(data_c)
centroids=kmeans.cluster_centers_
U=centroids
U=U.T




def euclidean_dist(X,Y):
    d=np.linalg.norm(X-Y)
    return d


epsilon=500000
hyp_par=0.1
V=np.zeros((p,n))
for i in range(p):
    for j in range(n):
        d=euclidean_dist(Y[:,j], U[:,i])
        if(d<epsilon):
            V[i,j]=np.exp(-hyp_par*d*d)





V_p=np.zeros((p,n))
for i in range(p):
    for j in range(n):
        d=euclidean_dist(Y[:,j], U[:,i])
        if(d<epsilon):
            V_p[i,j]=np.exp(-hyp_par*d)
       



Z=np.zeros((p,n))
Z_p=np.zeros((p,n))
for j in range(n):
    s=np.sum(V[:,j])
    s1=np.sum(V_p[:,j])
    for i in range(p):
        Z[i,j]=V[i,j]/s
        Z_p[i,j]=V_p[i,j]/s1




D=np.zeros((p,p))
D_p=np.zeros((p,p))
for i in range(p):
    s=np.sum(Z[i,:])
    s1=np.sum(Z[i,:])
    for j in range(p):
        if(i==j):
            D[i,j]=s**(-1/2)
            D_p[i,j]=s1**(-1/2)




Z=np.dot(D,Z)
Z_p=np.dot(D_p,Z_p)
matrix=np.dot(Z.T, Z)
matrix_p=np.dot(Z_p.T, Z_p)
n_clusters=k


print("Euclidean With Gaussian Kernel")
spectral_cluster=SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
cluster_labels=spectral_cluster.fit_predict(matrix)
ari=adjusted_rand_score(true_labels, cluster_labels)
nmi=normalized_mutual_info_score(true_labels,cluster_labels)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))


print("Euclidean With Poisson Kernel")
spectral_cluster=SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
cluster_labels_p=spectral_cluster.fit_predict(matrix_p)
ari=adjusted_rand_score(true_labels, cluster_labels_p)
nmi=normalized_mutual_info_score(true_labels,cluster_labels_p)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))



#############################################################################


def poincare_dist(X,Y):
    pq=np.linalg.norm(X-Y)
    p=np.linalg.norm(X)
    q=np.linalg.norm(Y)
    s=1+((2*(pq**2))/(1-(p**2))*(1-(q**2)))
    d=math.acosh(s)
    return d




epsilon=500000
hyp_par=0.1
V=np.zeros((p,n))
for i in range(p):
    for j in range(n):
        d=poincare_dist(X[:,j], U[:,i])
        if(d<epsilon):
            V[i,j]=np.exp(-hyp_par*d*d)





V_p=np.zeros((p,n))
for i in range(p):
    for j in range(n):
        d=poincare_dist(X[:,j], U[:,i])
        if(d<epsilon):
            V_p[i,j]=np.exp(-hyp_par*d)
       



Z=np.zeros((p,n))
Z_p=np.zeros((p,n))
for j in range(n):
    s=np.sum(V[:,j])
    s1=np.sum(V_p[:,j])
    for i in range(p):
        Z[i,j]=V[i,j]/s
        Z_p[i,j]=V_p[i,j]/s1




D=np.zeros((p,p))
D_p=np.zeros((p,p))
for i in range(p):
    s=np.sum(Z[i,:])
    s1=np.sum(Z[i,:])
    for j in range(p):
        if(i==j):
            D[i,j]=s**(-1/2)
            D_p[i,j]=s1**(-1/2)




Z=np.dot(D,Z)
Z_p=np.dot(D_p,Z_p)
matrix=np.dot(Z.T,Z)
matrix_p=np.dot(Z_p.T, Z_p)
n_clusters=k


print("Hyperbolic With Gaussian Kernel")
spectral_cluster=SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
cluster_labels=spectral_cluster.fit_predict(matrix)
ari=adjusted_rand_score(true_labels, cluster_labels)
nmi=normalized_mutual_info_score(true_labels,cluster_labels)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))


print("Hyperbolic With Poisson Kernel")
spectral_cluster=SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
cluster_labels_p=spectral_cluster.fit_predict(matrix_p)
ari=adjusted_rand_score(true_labels, cluster_labels_p)
nmi=normalized_mutual_info_score(true_labels,cluster_labels_p)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))









            
        


