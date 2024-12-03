import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv')

# Garantir que as colunas numéricas sejam tratadas corretamente
df['Faixa Remun Média (SM)'] = pd.to_numeric(df['Faixa Remun Média (SM)'], errors='coerce')
df['Faixa Hora Contrat'] = pd.to_numeric(df['Faixa Hora Contrat'], errors='coerce')
df['Escolaridade após 2005'] = pd.to_numeric(df['Escolaridade após 2005'], errors='coerce')

# Filtrar os dados

df_filtered = df[
    (~df['Faixa Remun Média (SM)'].isin([-1, 9, 99])) &
    (~df['Faixa Hora Contrat'].isin([-1, 9, 99])) &
    (~df['Escolaridade após 2005'].isin([-1, 9, 99]))
]

# Selecionar variáveis de interesse
df_selected = df_filtered[['Faixa Remun Média (SM)', 'Faixa Hora Contrat', 'Escolaridade após 2005']]
df_selected = df_selected.dropna()

# Agrupar salários em faixas predefinidas
faixas_salario = [0, 2, 5, 10, 15]  # Faixas de Salário em SM
df_selected['Faixa Salário (SM)'] = pd.cut(df_selected['Faixa Remun Média (SM)'], bins=faixas_salario, labels=['0-2', '2-5', '5-10', '10-15'])

# Agrupar escolaridade em categorias manuais (com base em valores comuns)
df_selected['Faixa Escolaridade'] = pd.cut(df_selected['Escolaridade após 2005'], bins=[0, 2, 5, 10], labels=['Baixa', 'Média', 'Alta'])

# Analisar correlação entre variáveis
correlation_matrix = df_selected[['Faixa Remun Média (SM)', 'Faixa Hora Contrat', 'Escolaridade após 2005']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

# Estatísticas descritivas por faixa de salário
print("\nEstatísticas descritivas por faixa de salário:")
print(df_selected.groupby('Faixa Salário (SM)').describe())

# Contagem de registros por faixa de escolaridade
print("\nDistribuição de trabalhadores por faixa de escolaridade:")
print(df_selected['Faixa Escolaridade'].value_counts())

# Gráfico: Distribuição por Faixa Salarial e Escolaridade
plt.figure(figsize=(10, 6))
sns.countplot(data=df_selected, x='Faixa Salário (SM)', hue='Faixa Escolaridade', palette='viridis')
plt.title('Distribuição de Trabalhadores por Faixa Salarial e Escolaridade')
plt.xlabel('Faixa Salário (SM)')
plt.ylabel('Contagem')
plt.legend(title='Escolaridade')
plt.show()

# Exemplo de análise cruzada
print("\nMédia de horas contratadas por faixa salarial e escolaridade:")
print(df_selected.groupby(['Faixa Salário (SM)', 'Faixa Escolaridade'])['Faixa Hora Contrat'].mean())
