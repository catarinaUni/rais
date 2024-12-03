import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Carregar os dados
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv', low_memory=False)

# Converter a coluna 'Idade' para numérico e tratar valores inválidos
df['Idade'] = pd.to_numeric(df['Idade'], errors='coerce')

# Filtrar valores inválidos para 'Causa Afastamento 1' (remover 99 e -1)
df_filtered = df.loc[~df['Causa Afastamento 1'].isin([99, -1])]

# Filtrar valores inválidos para 'Idade'
df_filtered = df_filtered.loc[~df_filtered['Idade'].isin([99, 9])]

# Selecionar as colunas de interesse
df_selected = df_filtered[['Causa Afastamento 1', 'Idade']].copy()

# Criar faixas de idade
bins = [0, 18, 30, 40, 50, 60, 100]
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '60+']
df_selected['Faixa Idade'] = pd.cut(df_selected['Idade'], bins=bins, labels=labels, right=False)

# Criar novas colunas categorizadas para 'Causa Afastamento 1' e 'Faixa Idade'
df_selected['Causa Afastamento 1 (Categoria)'] = df_selected['Causa Afastamento 1'].astype(str)
df_selected['Faixa Idade (Categoria)'] = df_selected['Faixa Idade'].astype(str)

# Criar transações
transactions = df_selected[['Causa Afastamento 1 (Categoria)', 'Faixa Idade (Categoria)']]

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
