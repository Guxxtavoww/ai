import pandas as pd
from keras import Sequential, layers
from sklearn.model_selection import train_test_split

classe = pd.read_csv('./saidas_breast.csv');
previsores = pd.read_csv('./entradas_breast.csv');

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25);
 
classfyer = Sequential();
classfyer.add(layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30));
classfyer.add(layers.Dense(units=1, activation='sigmoid'));
classfyer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy']);
classfyer.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)
