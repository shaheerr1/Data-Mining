import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

data = pd.read_csv(r'C:\datasets\Processed_Encoded_data_with_split.csv')


features = data[['Total', 'Unit price', 'Quantity']]


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

data['Cluster'] = clusters

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.scatter(features_scaled[:, 0], features_scaled[:, 1],
            c=clusters, cmap='viridis', label='clusters')
plt.title('Total vs Unit Price')
plt.xlabel('Total (standardized)')
plt.ylabel('Unit Price (standardized)')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(features_scaled[:, 0], features_scaled[:, 2],
            c=clusters, cmap='viridis', label='clusters')
plt.title('Total vs Quantity')
plt.xlabel('Total (standardized)')
plt.ylabel('Quantity (standardized)')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
