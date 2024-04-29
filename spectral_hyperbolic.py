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

#Encoding the label column 
#For example, the wisc dataset contained the labels in the column indexed by 9. 
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

#Y is the original Dataset, X is the embedded dataset
#Shape of the Data Matrix, m=no of rows, n=no of columns with rows representing data points


######################################################################################################################################################################
#Euclidean Spectral Clustering with Gaussian and Poisson Kernels

#Defining the Euclidean Distance
def euclidean_dist(X,Y):
    d=np.linalg.norm(X-Y)
    return d



E=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        E[i][j]=euclidean_dist(Y[i,:],Y[j,:])


epsilon=500000 #Cut off length, has been set high to make the graph connected as much as possible

#Setting up the hyper-parameter sigma
hyp_par=0.1

#Affinity Matrices
Aff=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if(E[i][j]<epsilon):
            Aff[i][j]=np.exp(-hyp_par*E[i][j]*E[i][j])
        else:
            Aff[i][j]=0
        




Aff_p=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if(E[i][j]<epsilon):
            Aff_p[i][j]=np.exp(-2*hyp_par*E[i][j])
        else:
            Aff_p[i][j]=0

            

#Degree Matrices
D_bar=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if(i==j):
            D_bar[i][j]=np.sum(Aff[i,:])
        else:
            D_bar[i][j]=0




D_bar_p=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if(i==j):
            D_bar_p[i][j]=np.sum(Aff_p[i,:])
        else:
            D_bar_p[i][j]=0


n_clusters=k
#For the ESCA with Gaussian Kernel
L_bar=np.subtract(D_bar,Aff)
eigenvalues,eigenvectors=np.linalg.eigh(L_bar)
PQ=eigenvectors[:,1:n_clusters+1]
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(PQ)
cluster_labels=kmeans.labels_


silhouette = silhouette_score(Y, cluster_labels)
davies_bouldin = davies_bouldin_score(Y, cluster_labels)
calinski_harabasz = calinski_harabasz_score(Y, cluster_labels)

print("Euclidean Spectral Clustering with Gaussian Kernel")
print("Silhouette Score={}\n".format(silhouette))
print("Davies Bouldin Score={}\n".format(davies_bouldin))
print("Calinski Harabasz Index={}\n".format(calinski_harabasz))
ari=adjusted_rand_score(true_labels, cluster_labels)
nmi=normalized_mutual_info_score(true_labels,cluster_labels)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))
print("Cluster Assignments(for Euclidean Spectral Clustering):", cluster_labels)



#For the ESCA with Poisson Kernel
L_bar_p=np.subtract(D_bar_p,Aff_p)
eigenvalues,eigenvectors=np.linalg.eigh(L_bar_p)
PQ=eigenvectors[:,1:n_clusters+1]
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(PQ)
cluster_labels_p=kmeans.labels_


silhouette = silhouette_score(Y, cluster_labels_p)
davies_bouldin = davies_bouldin_score(Y, cluster_labels_p)
calinski_harabasz = calinski_harabasz_score(Y, cluster_labels_p)
print("Euclidean Spectral Clustering with Poisson Kernel")
print("Silhouette Score={}\n".format(silhouette))
print("Davies Bouldin Score={}\n".format(davies_bouldin))
print("Calinski Harabasz Index={}\n".format(calinski_harabasz))
ari=adjusted_rand_score(true_labels, cluster_labels_p)
nmi=normalized_mutual_info_score(true_labels,cluster_labels_p)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))
print("Cluster Assignments(for Euclidean Spectral Clustering):", cluster_labels_p)


###########################################################################################################################################################################
#For Hyperbolic Spectral Clustering with Gaussian and Poisson Kernels


#Defining the Poincare Distance
def poincare_dist(X,Y):
    pq=np.linalg.norm(X-Y)
    p=np.linalg.norm(X)
    q=np.linalg.norm(Y)
    s=1+((2*(pq**2))/(1-(p**2))*(1-(q**2)))
    d=math.acosh(s)
    return d


#R is the matrix containing the pairwise distances of the embedded data points 
R=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        R[i][j]=poincare_dist(X[i,:],X[j,:])


W=np.zeros((m,m))
for i in range(m):
     for j in range(m):
          if(R[i][j]<epsilon):
               W[i][j]=np.exp(-hyp_par*R[i][j]*R[i][j])
          else:
               W[i][j]=0






W_p=np.zeros((m,m))
for i in range(m):
     for j in range(m):
          if(R[i][j]<epsilon):
               W_p[i][j]=np.exp(-2*hyp_par*R[i][j])
          else:
               W_p[i][j]=0



#Applying the Spectral CLustering on the affinity matrices as per HSCA
spectral_cluster=SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
cluster_labels_h=spectral_cluster.fit_predict(W)
cluster_labels_h_p=spectral_cluster.fit_predict(W_p)



######################################################################################################################################################################
#Defining Mobius Addition and Mobius Scalar Multiplication
def mob_add(x,y):
    x_n=np.linalg.norm(x)
    y_n=np.linalg.norm(y)
    s=((1+2*np.dot(x,y)+y_n**2)*x+(1-x_n**2)*y)/(1+2*(np.dot(x,y)+(x_n*y_n)**2))
    return s

def mob_sc_mult(c,x):
    x_n=np.linalg.norm(x)
    s=math.atanh(x_n)
    s=c*s
    p=(math.tanh(s))*(1/x_n)
    q=p*x
    return q




###################################################################################################################################################################
#Defining Silhoutte Score with the Poincare Distance matrix as input

def silhouette_sample_score_hyp(X, labels, distances):
    """
    Compute Silhouette Sample Score using Poincare distance as a metric=distances.
    
    Parameters:
        X (array-like): The input data samples.
        labels (array-like): Predicted cluster labels for each sample.
        
    Returns:
        sil_samples (array-like): Silhouette Sample Scores for each sample.
    """
    n_samples = len(X)
    sil_samples = np.zeros(n_samples)
    
    
    
    for i in range(n_samples):
        cluster_label = labels[i]
        cluster_distances = distances[i][labels == cluster_label]
        a = np.mean(cluster_distances)
        
        # Calculate nearest cluster
        other_labels = np.unique(labels[labels != cluster_label])
        b = np.min([np.mean(distances[i][labels == other_label]) for other_label in other_labels])
        
        sil_samples[i] = (b - a) / max(a, b)
    
    return sil_samples

#Defining the Davies-Bouldin Score with Poincare Metric
def davies_bouldin_score_hyp(X, labels, metric_hyp):
    """
    Compute Davies-Bouldin Score using a specified metric.
    
    Parameters:
        X (array-like): The input data samples.
        labels (array-like): Predicted cluster labels for each sample.
        metric (str): The distance metric to use (default is 'euclidean').
        
    Returns:
        float: Davies-Bouldin Score for the clustering result.
    """
    # Number of clusters
    k=len(np.unique(labels))
    
    # Calculate centroid for each cluster
    centroids = np.array([np.mean(X[labels == i], axis=0) for i in range(k)])
    """centroids=np.zeros((k,X.shape[1]))
    for i in range(k):
        r_xx=np.zeros(X.shape[1])
        p=0
        for j in range(len(X)):
            if(labels[j]==i):
                r_x=X[j]
                r_xx=mob_add(r_xx,r_x)
                p=p+1
        r_xx=mob_sc_mult((1/p),r_xx)
        centroids[i]=r_xx"""
    
    #centroids = np.array([np.mean(X[labels == i], axis=0) for i in range(k)])
    
    # Compute intra-cluster distances
    intra_cluster_distances = np.zeros(k)
    for i in range(k):
        cluster_points = X[labels == i]
        t=cluster_points.shape[0]
        r=np.zeros(t)
        j=0
        for points in cluster_points:
            r[j]=metric_hyp(points,centroids[i])
            j=j+1

        intra_cluster_distances[i]=np.mean(r)    
        
    
    # Compute pairwise distances between centroids
    
    centroid_distances = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            centroid_distances[i,j]=metric_hyp(centroids[i],centroids[j])
            
    # Calculate Davies-Bouldin score
    db_scores = np.zeros(k)
    for i in range(k):
        # Exclude the current cluster's centroid from the calculation
        db_scores[i] = np.max([(intra_cluster_distances[i] + intra_cluster_distances[j]) / centroid_distances[i][j]
                               for j in range(k) if j != i])
    
    return np.mean(db_scores)






#Defining the Calinski Harabasz Score with the Poincare Metric
def compute_calinski_harabasz_score(X, n_clusters, metric):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    score = metrics.calinski_harabasz_score(X, labels, metric=metric)
    return score





ss=silhouette_sample_score_hyp(X,cluster_labels_h, R)
silhouette = np.mean(ss)
print("Hyperbolic Spectral Clustering with Gaussian Kernels")
davies_bouldin = davies_bouldin_score_hyp(X, cluster_labels_h, poincare_dist)
calinski_harabasz = calinski_harabasz_score(X, cluster_labels_h)
print("Silhouette Score={}\n".format(silhouette))
print("Davies Bouldin Score={}\n".format(davies_bouldin))
print("Calinski Harabasz Score={}\n".format(calinski_harabasz))
ari=adjusted_rand_score(true_labels, cluster_labels_h)
nmi=normalized_mutual_info_score(true_labels,cluster_labels_h)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))
print("Cluster Assignments(for Hyperbolic Spectral Clustering):", cluster_labels_h)


ss=silhouette_sample_score_hyp(X,cluster_labels_h_p, R)
silhouette = np.mean(ss)
print("Hyperbolic Spectral Clustering with Poisson Kernel")
davies_bouldin = davies_bouldin_score_hyp(X, cluster_labels_h_p, poincare_dist)
calinski_harabasz = calinski_harabasz_score(X, cluster_labels_h_p)
print("Silhouette Score={}\n".format(silhouette))
print("Davies Bouldin Score={}\n".format(davies_bouldin))
print("Calinski Harabasz Score={}\n".format(calinski_harabasz))
ari=adjusted_rand_score(true_labels, cluster_labels_h_p)
nmi=normalized_mutual_info_score(true_labels,cluster_labels_h_p)
print("Adjusted Rand Score={}\n".format(ari))
print("Normalized Mutual Information={}\n".format(nmi))
print("Cluster Assignments(for Hyperbolic Spectral Clustering):", cluster_labels_h_p)



#########################################################################################################################################################################
#Plotting the t-SNE visualization of the dataset

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(Y)



#Visualizing the T-SNE representation of the entire dataset
plt.scatter(X_2d[:,0], X_2d[:,1],c=true_labels, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization for the dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.subplots()


# Visualizing the Hyperbolic Spectral CLustering with Gaussian Kernel
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels_h, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization of Hyperbolic Clusters_Gaussian')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.subplots()

# Visualizing the Hyperbolic Spectral CLustering with Poisson Kernel
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels_h_p, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization of Hyperbolic Clusters_Poisson')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.subplots()


#Visualizing the Euclidean Spectral CLustering with Gaussian Kernel
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization of Euclidean Clusters_Gaussian')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.subplots()



#Visualizing the Euclidean Spectral CLustering with Poisson Kernel
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels_p, cmap='viridis', edgecolors='k')
plt.title('t-SNE Visualization of Euclidean Clusters_Poisson')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()































