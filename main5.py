from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv', low_memory=False)

# Substituir valores 99 por NaN nas colunas selecionadas
df[['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005']] = df[['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005']].replace(99, np.nan)
df[['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005']] = df[['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005']].replace(999, np.nan)

# Substituir valores 99 por NaN na coluna 'Faixa Remun Média (SM)'
df['Faixa Remun Média (SM)'] = df['Faixa Remun Média (SM)'].replace(99, np.nan)

# Remover as linhas que contêm NaN nas variáveis selecionadas
df = df.dropna(subset=['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005', 'Faixa Remun Média (SM)'])

# Ajustar as variáveis para as faixas
# Faixa Tempo Emprego - de 1 à 8
df['Faixa Tempo Emprego'] = df['Faixa Tempo Emprego'].apply(lambda x: x if x >= 1 and x <= 8 else np.nan)

# Faixa Remun Média (SM) - de 0 à 12
df['Faixa Remun Média (SM)'] = df['Faixa Remun Média (SM)'].apply(lambda x: x if x >= 0 and x <= 12 else np.nan)

# Faixa Hora Contrat - de 1 à 6
df['Faixa Hora Contrat'] = df['Faixa Hora Contrat'].apply(lambda x: x if x >= 1 and x <= 6 else np.nan)

# Remover as linhas que contêm NaN após aplicar as faixas
df = df.dropna(subset=['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005', 'Faixa Remun Média (SM)'])

# Selecionar variáveis independentes e a variável alvo
X = df[['Faixa Hora Contrat', 'Faixa Tempo Emprego', 'Idade', 'Escolaridade após 2005']]
y = df['Faixa Remun Média (SM)']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Coeficientes das variáveis
print("Coeficientes:", model.coef_)

# Previsões do modelo
y_pred = model.predict(X_test)

# Gráficos de dispersão das variáveis independentes com a variável dependente
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 'Faixa Hora Contrat' vs 'Faixa Remun Média (SM)'
axs[0, 0].scatter(X_test['Faixa Hora Contrat'], y_test, color='blue', label='Real')
axs[0, 0].scatter(X_test['Faixa Hora Contrat'], y_pred, color='red', label='Predito')
axs[0, 0].set_title('Faixa Hora Contrat vs Faixa Remun Média (SM)')
axs[0, 0].set_xlabel('Faixa Hora Contrat')
axs[0, 0].set_ylabel('Faixa Remun Média (SM)')
axs[0, 0].legend()

# 'Faixa Tempo Emprego' vs 'Faixa Remun Média (SM)'
axs[0, 1].scatter(X_test['Faixa Tempo Emprego'], y_test, color='blue', label='Real')
axs[0, 1].scatter(X_test['Faixa Tempo Emprego'], y_pred, color='red', label='Predito')
axs[0, 1].set_title('Faixa Tempo Emprego vs Faixa Remun Média (SM)')
axs[0, 1].set_xlabel('Faixa Tempo Emprego')
axs[0, 1].set_ylabel('Faixa Remun Média (SM)')
axs[0, 1].legend()

# 'Idade' vs 'Faixa Remun Média (SM)'
axs[1, 0].scatter(X_test['Idade'], y_test, color='blue', label='Real')
axs[1, 0].scatter(X_test['Idade'], y_pred, color='red', label='Predito')
axs[1, 0].set_title('Idade vs Faixa Remun Média (SM)')
axs[1, 0].set_xlabel('Idade')
axs[1, 0].set_ylabel('Faixa Remun Média (SM)')
axs[1, 0].legend()

# 'Escolaridade após 2005' vs 'Faixa Remun Média (SM)'
axs[1, 1].scatter(X_test['Escolaridade após 2005'], y_test, color='blue', label='Real')
axs[1, 1].scatter(X_test['Escolaridade após 2005'], y_pred, color='red', label='Predito')
axs[1, 1].set_title('Escolaridade após 2005 vs Faixa Remun Média (SM)')
axs[1, 1].set_xlabel('Escolaridade após 2005')
axs[1, 1].set_ylabel('Faixa Remun Média (SM)')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Gráficos completos de todas as variáveis com faixas
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 'Faixa Hora Contrat' vs 'Faixa Remun Média (SM)'
axs[0, 0].scatter(df['Faixa Hora Contrat'], df['Faixa Remun Média (SM)'], color='blue')
axs[0, 0].set_title('Faixa Hora Contrat vs Faixa Remun Média (SM)')
axs[0, 0].set_xlabel('Faixa Hora Contrat')
axs[0, 0].set_ylabel('Faixa Remun Média (SM)')

# 'Faixa Tempo Emprego' vs 'Faixa Remun Média (SM)'
axs[0, 1].scatter(df['Faixa Tempo Emprego'], df['Faixa Remun Média (SM)'], color='blue')
axs[0, 1].set_title('Faixa Tempo Emprego vs Faixa Remun Média (SM)')
axs[0, 1].set_xlabel('Faixa Tempo Emprego')
axs[0, 1].set_ylabel('Faixa Remun Média (SM)')

# 'Idade' vs 'Faixa Remun Média (SM)'
axs[1, 0].scatter(df['Idade'], df['Faixa Remun Média (SM)'], color='blue')
axs[1, 0].set_title('Idade vs Faixa Remun Média (SM)')
axs[1, 0].set_xlabel('Idade')
axs[1, 0].set_ylabel('Faixa Remun Média (SM)')

# 'Escolaridade após 2005' vs 'Faixa Remun Média (SM)'
axs[1, 1].scatter(df['Escolaridade após 2005'], df['Faixa Remun Média (SM)'], color='blue')
axs[1, 1].set_title('Escolaridade após 2005 vs Faixa Remun Média (SM)')
axs[1, 1].set_xlabel('Escolaridade após 2005')
axs[1, 1].set_ylabel('Faixa Remun Média (SM)')

plt.tight_layout()
plt.show()
