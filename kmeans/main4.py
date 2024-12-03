from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar dados
df = pd.read_csv('RAIS_VINC_PUB_NORDESTE.csv', low_memory=False)

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
