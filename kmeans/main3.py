import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Carregar os dados
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv', low_memory=False)

# Converter colunas para numérico e tratar valores inválidos
df['Faixa Remun Média (SM)'] = pd.to_numeric(df['Faixa Remun Média (SM)'], errors='coerce')
df['Faixa Hora Contrat'] = pd.to_numeric(df['Faixa Hora Contrat'], errors='coerce')

# Filtrar valores inválidos
df_filtered = df.loc[~df['Faixa Remun Média (SM)'].isin([99, 9])]
df_filtered = df_filtered.loc[~df_filtered['Faixa Hora Contrat'].isin([99, 9])]

# Selecionar colunas de interesse
df_selected = df_filtered[['Faixa Remun Média (SM)', 'Faixa Hora Contrat']].copy()

# Criar novas colunas categorizadas
df_selected['Faixa Remun Média (Categoria)'] = 'Rem_' + df_selected['Faixa Remun Média (SM)'].astype(str)
df_selected['Faixa Hora Contrat (Categoria)'] = 'Horas_' + df_selected['Faixa Hora Contrat'].astype(str)

# Criar transações
transactions = df_selected[['Faixa Remun Média (Categoria)', 'Faixa Hora Contrat (Categoria)']]

# Converter para One-Hot Encoding
one_hot = pd.get_dummies(transactions)

# Aplicar Apriori
frequent_itemsets = apriori(one_hot, min_support=0.1, use_colnames=True)

# Gerar regras de associação
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6, num_itemsets=len(frequent_itemsets))

# Exibir resultados
print("\nConjuntos Frequentes:")
print(frequent_itemsets)

print("\nRegras de Associação:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])