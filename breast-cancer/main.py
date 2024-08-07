import pandas as pd
from sklearn.model_selection import train_test_split

previsores = pd.read_csv('./entradas_breast.csv')
classes = pd.read_csv('./saidas_breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classes, test_size=0.25);

print(previsores_treinamento)