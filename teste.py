#Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    #Ler ambos csv (Fake_News) 
    df_true = pd.read_csv('data/True.csv')
    df_false = pd.read_csv('data/Fake.csv')

    print(df_true.head())

    print(df_false.head())

    #classe -> se 0, então a notícia é verdadeira; se 1, então a notícia é fake
    #vamos adicionar coluna classe
    df_true['classe'] = 0
    #df_True = df_True.assign(classe=0)
    df_false['classe'] = 1
    #df_False = df_False.assign(classe=1)

    #vamos juntar ambas databases :)
    # Une df1 e df2 usando a coluna 'id' como referência
    df_merged = pd.merge(df_true, df_false, on='id', how='inner')
    df_merged.head()
    #assim, obtemos a base de dados que iremos utilizar

    #valores nulos
    df_merged.isna().values.any()

    #vamos dividir os dados por teste (70%) e treino(30%)
    # X são as tuas variáveis de entrada (features)
    # y é o que queres prever (target/label)
    X = df_merged.drop('classe', axis=1) 
    y = df_merged['classe']

    # Divide os dados (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




if __name__ == "__main__":
    main()