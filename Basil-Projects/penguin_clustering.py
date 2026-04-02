import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
#fills the missing values with the mean of the column
from sklearn.impute import SimpleImputer
#transforms each feature so it has a mean of 0 and standard deviation of 1.
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Principle Component Analysis is a dimensionality reduction technique that transforms the data into a new coordinate system, where the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
from sklearn.decomposition import PCA
# Agglomerative Clustering is a hierarchical clustering method that builds nested clusters by merging or splitting them successively. It starts with each data point as its own cluster and then merges the closest pairs of clusters until only one cluster (or a specified number of clusters) remains.
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
# Adjusted Rand Index (ARI) is a measure of the similarity between two clusterings, adjusted for chance. It ranges from -1 to 1, where 1 indicates perfect agreement between the clusterings, 0 indicates random labeling, and negative values indicate less agreement than expected by chance.
from sklearn.metrics import adjusted_rand_score

df=pd.read_csv("penguins.csv")
pair_df=df[["species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
pair_df=pair_df.dropna()
sns.pairplot(pair_df,hue="species")
plt.show()
x=df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
sns.heatmap(x.corr(),annot=True)
y_true=df["species"]
plt.show()
# filling the missing values with the mean of the column
x_imputed=SimpleImputer(strategy="mean").fit_transform(x)
x_scaled=StandardScaler().fit_transform(x_imputed)
inertias=[]
k_range=range(2,9)
silhouette_scores=[]
for k in k_range:
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(x_scaled)
    silhouette_scores.append(silhouette_score(x_scaled,kmeans.labels_))
    inertias.append(kmeans.inertia_)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Elbow plot
axes[0].plot(k_range, inertias, "o-")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

# Silhouette plot
axes[1].plot(k_range, silhouette_scores, "s-")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")

plt.tight_layout()
plt.show()
kmeans=KMeans(n_clusters=3,random_state=42,n_init=10)
cluster_labels=kmeans.fit_predict(x_scaled)
df["Cluster"]=cluster_labels
print(df["Cluster"].value_counts())
print(df.groupby("Cluster")[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']].mean())
pca=PCA(n_components=2,random_state=42)
x_pca=pca.fit_transform(x_scaled)
print(pca.explained_variance_ratio_)
# Encode species labels for coloring the scatter plot
species_encoded = pd.Categorical(df["species"]).codes
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=species_encoded, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters (PCA projection)")
plt.colorbar(label="Species")
plt.show()
sample_idx = py.random.choice(len(x_scaled), size=80, replace=False)
X_sample = x_scaled[sample_idx]
z=linkage(X_sample, method="ward")
dendrogram(z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
aggl=AgglomerativeClustering(n_clusters=3)
hier_labels=aggl.fit_predict(x_scaled)
sil = silhouette_score(x_scaled, cluster_labels)
print(f"Silhouette Score: {sil:.4f}")
# Calculate the Adjusted Rand Index (ARI) between the true species labels and the cluster labels
ari = adjusted_rand_score(species_encoded, cluster_labels)
print(f"Adjusted Rand Index: {ari:.4f}")

ct = pd.crosstab(df["species"], df["Cluster"], 
                 rownames=["True Species"], 
                 colnames=["Predicted Cluster"])
print(ct)
sns.heatmap(ct, annot=True, fmt="d")
plt.show()