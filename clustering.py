import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

encoded_data = pd.read_csv(r'C:\datasets\Encoded_data.csv')

numeric_and_boolean_columns = encoded_data.select_dtypes(
    include=[np.number, 'bool']).columns.tolist()

non_numeric_non_boolean_columns = encoded_data.select_dtypes(
    exclude=[np.number, 'bool']).columns.tolist()

print("Numeric and boolean columns = ", numeric_and_boolean_columns)
print("\n")
print("Non numeric and non boolean columns = ", non_numeric_non_boolean_columns)


encoded_data['Datetime'] = pd.to_datetime(encoded_data['Datetime'])
encoded_data.set_index('Datetime', inplace=True)

clustering_data = encoded_data[numeric_and_boolean_columns]

kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(clustering_data)
encoded_data['Cluster'] = kmeans.labels_

cluster_monthly_stats = encoded_data.groupby([pd.Grouper(freq='M'), 'Cluster'])[
    numeric_and_boolean_columns].mean()

print(cluster_monthly_stats.head())

# -------------------------------------

cluster_counts = encoded_data.groupby(
    [pd.Grouper(freq='M'), 'Cluster']).size().unstack(fill_value=0)
cluster_counts.plot(kind='bar', stacked=True)
plt.title('Cluster Count per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Cluster')
plt.xticks(rotation=45)
plt.show()

# -------------------------------------

plt.scatter(encoded_data['Total'], encoded_data['Quantity'],
            c=encoded_data['Cluster'], cmap='viridis')
plt.title('Clusters based on Total and Quantity')
plt.xlabel('Total')
plt.ylabel('Quantity')
plt.colorbar(label='Cluster')
plt.show()

# -------------------------------------

for column in numeric_and_boolean_columns:
    if column not in ['Cluster', 'Other Non-relevant Columns']:
        plt.figure(figsize=(10, 6))
        for cluster in cluster_monthly_stats[column].unstack(level=1).columns:
            cluster_data = cluster_monthly_stats[column].unstack(level=1)[
                cluster]
            cluster_data.plot(label=f'Cluster {cluster}')
        plt.title(f'Trend for {column} over Time')
        plt.xlabel('Month')
        plt.ylabel(column)
        plt.legend()
        plt.show()
