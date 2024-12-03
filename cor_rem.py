import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Passo 1: Carregar o arquivo
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv')

# Exibe as primeiras linhas para verificar os dados
print(df.head())

# Passo 2: Seleção das variáveis de interesse
df_selected = df[['Faixa Remun Média (SM)', 'Idade']]  # Selecionando as variáveis de remuneração e raça/cor
df_selected = df_selected.dropna()  # Remover valores ausentes

# Passo 3: Convertendo variáveis categóricas para variáveis dummy
df_selected = pd.get_dummies(df_selected, drop_first=True)  # Converte a variável 'Raça Cor' para dummy

# Passo 4: Normalização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Passo 5: Aplicação do KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Passo 6: Visualização dos clusters diretamente
plt.figure(figsize=(8,6))
plt.scatter(df_selected['Faixa Remun Média (SM)'], df_selected['Idade'], c=df['Cluster'], cmap='viridis')
plt.title('KMeans Clustering - Resultados')
plt.xlabel('Faixa Remun Média (SM)')
plt.colorbar(label='Cluster')
plt.show()

# Passo 7: Exibir as estatísticas por cluster (média das variáveis)
print("\nEstatísticas por cluster:")
print(df.groupby('Cluster')[['Faixa Remun Média (SM)', 'Idade']].mean())

# Passo 8: Exibir o número de trabalhadores em cada cluster
print("\nNúmero de trabalhadores em cada cluster:")
print(df.groupby('Cluster').size())

# Passo 9: Exibir os centroides dos clusters
print("\nCentroides dos clusters:")
print(kmeans.cluster_centers_)
