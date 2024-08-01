
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

encoded_data = pd.read_csv(
    r'C:\datasets\Processed_Encoded_data_with_split.csv')

encoded_data['Datetime'] = pd.to_datetime(encoded_data['Datetime'])
encoded_data.set_index('Datetime', inplace=True)

if 'Cluster' not in encoded_data.columns:
    clustering_data = encoded_data[['Total', 'Quantity', 'Rating']]
    kmeans = KMeans(n_clusters=3, n_init=10,
                    random_state=0).fit(clustering_data)
    encoded_data['Cluster'] = kmeans.labels_


cluster_evolution_simple = encoded_data.groupby(
    [pd.Grouper(freq='M'), 'Cluster']).size().unstack(fill_value=0)

print(cluster_evolution_simple)


# -------------------------------------

cluster_evolution_simple.plot(kind='bar', stacked=True)
plt.title('Cluster Evolution Over Time')
plt.ylabel('Number of Entries')
plt.xlabel('Month')
plt.legend(title='Cluster')
plt.xticks(rotation=45)
plt.show()


# -------------------------------------


plt.scatter(encoded_data['Total'], encoded_data['Quantity'],
            c=encoded_data['Cluster'], cmap='viridis')
plt.title('KMeans Clusters Visualization')
plt.xlabel('Total')
plt.ylabel('Quantity')
plt.colorbar(label='Cluster')
plt.show()
