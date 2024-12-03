import pandas as pd

# Passo 1: Carregar os dados
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv')

# Passo 2: Filtrar por "Escolaridade após 2005" até o nível 7
df_filtrado = df[(df['Faixa Remun Média (SM)'] >= 0) & (df['Faixa Remun Média (SM)'] <= 11)]
df_filtrado = df_filtrado[df_filtrado['Faixa Hora Contrat'] == 1]

# Passo 3: Visualizar os dados filtrados
print(df_filtrado)

# (Opcional) Salvar os dados filtrados em um novo arquivo CSV
df_filtrado.to_csv('dados_filtrados.csv', index=False)
