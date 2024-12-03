import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv')

df['Faixa Remun Média (SM)'] = pd.to_numeric(df['Faixa Remun Média (SM)'], errors='coerce')
df['Faixa Hora Contrat'] = pd.to_numeric(df['Faixa Hora Contrat'], errors='coerce')

df_filtered = df[
    (~df['Faixa Remun Média (SM)'].isin([-1, 9, 99])) &
    (~df['Faixa Hora Contrat'].isin([-1, 9, 99]))
]



df_selected = df_filtered[['Faixa Remun Média (SM)', 'Faixa Hora Contrat']]

df_selected = df_selected.dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

kmeans = KMeans(n_clusters=5, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(df_scaled)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df_filtered['Cluster']

plt.figure(figsize=(8,6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis')
plt.title('KMeans Clustering - PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

print("\nOs 3 valores mais frequentes por cluster para 'Faixa Remun Média (SM)':")
print(df_filtered.groupby('Cluster')['Faixa Remun Média (SM)'].apply(lambda x: x.mode().head(3)))

print("\nOs 3 valores mais frequentes por cluster para 'Faixa Hora Contrat':")
print(df_filtered.groupby('Cluster')['Faixa Hora Contrat'].apply(lambda x: x.mode().head(3)))

print("\nNúmero de trabalhadores em cada cluster:")
print(df_filtered.groupby('Cluster').size())

print("\nCentroides dos clusters:")
print(kmeans.cluster_centers_)

print("\nDetalhes de cada cluster:")

for cluster_id in range(kmeans.n_clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_data = df_filtered[df_filtered['Cluster'] == cluster_id]
    
    cluster_data_cleaned = cluster_data[~cluster_data['Faixa Remun Média (SM)'].isin([99])]
    cluster_data_cleaned = cluster_data_cleaned[~cluster_data_cleaned['Faixa Hora Contrat'].isin([99])]
    
    if cluster_data_cleaned.empty:
        print("Este cluster contém apenas valores '99' ou não tem dados após filtragem.")
    else:
        print("\nEstatísticas descritivas (Faixa Remun Média (SM) e Faixa Hora Contrat) sem '99':")
        print(cluster_data_cleaned[['Faixa Remun Média (SM)', 'Faixa Hora Contrat']].describe())
        
        print("\nValores mais frequentes para 'Faixa Remun Média (SM)':")
        print(cluster_data_cleaned['Faixa Remun Média (SM)'].mode().head(3))
        
        print("\nValores mais frequentes para 'Faixa Hora Contrat':")
        print(cluster_data_cleaned['Faixa Hora Contrat'].mode().head(3))
        
        print("\nExemplo de observações dentro do cluster (sem '99'):")
        print(cluster_data_cleaned[['Faixa Remun Média (SM)', 'Faixa Hora Contrat']].head(5))
