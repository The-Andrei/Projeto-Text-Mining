#Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fazer download dos pacotes necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Conta a quantidade de palavras em maiusculas
def count_caps_letters(text):
    # Proteção contra valores nulos/NaNs
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return sum(1 for char in text if char.isupper())

def clean_text(text):
        #mantém apenas letras (a-z, A-Z), espaços e números, remove o resto. Coloca tudo em minusculas
        text = re.sub(r'[^\w\s]', '', text.lower()) 
        return text

def remove_stopwords_and_lemmatize(tokens):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Remove stopwords e aplica lematização
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
def main():
    #Ler ambos csvs
    df_true = pd.read_csv('data/True.csv')
    df_false = pd.read_csv('data/Fake.csv')

    #classe -> se 0, então a notícia é verdadeira; se 1, então a notícia é fake
    #vamos adicionar coluna classe
    df_true['classe'] = 0
    df_false['classe'] = 1

    #vamos juntar ambas databases :)
    df_total = pd.concat([df_true, df_false], ignore_index=True)
    print(df_total.info())

    #randomizing a database para as linhas com diferentes classes estarem mistruradas
    df_total = df_total.sample(frac=1).reset_index(drop=True)
    # print(df_total.head())

    # export dataframe to csv
    df_total.to_csv('data/total.csv', index=False)

    # ETAPA 1 (o que diz no guião):
        # Pré-processamento e Preparação
            # • Limpeza de Documentos: Implementar estratégias de preparação e limpeza adequadas ao tipo de texto;
            # • Extração de Propriedades: Aplicar técnicas de extração de característica relevantes;
            # • Estratégias de Ponderação: Implementar e justificar estratégias de ponderação de termos (TF-IDF, etc.);

    #normalização -> retirar caps, retirar caracteres especiais,

    #pergunta: deveriamos retirar caps e simbolos com ! sendo que pode existir uma correlação entre estes e noticias falsas
        # RESPOSTA:
        # temos de fazer o "mapeamento" de palvras e caracteres antes da limpeza. ex: "!!! : "entusiasmo" " - podemos fazer um dicionario
        # sim... a certo ponto temos de retirar a pontuação e etc...8

    #Extração: Crie colunas novas no seu DataFrame (ex: count_exclamation, is_all_caps).
    
    # Conta todos os "!" no texto e no título
    df_total['count_exclamation_title'] = df_total['title'].apply(lambda x: x.count('!'))
    df_total['count_exclamation_text'] = df_total['text'].apply(lambda x: x.count('!'))
    
    # Conta a quantidade de palavras em maiusculas no título e no texto
    df_total['count_caps_letters_title'] = df_total['title'].apply(count_caps_letters)
    df_total['count_caps_letters_text'] = df_total['text'].apply(count_caps_letters)

    ###### FALTA

    # Configura o pandas para não esconder colunas
    pd.set_option('display.max_columns', None)

    #Ver todas as colunas 
    print(df_total.head())

    #remover pontuação. caracteres especiais e colocar tudo em minusculas
    print("A limpar o texto...")
    df_total['text_clean'] = df_total['text'].apply(clean_text)
    
    #tokenizar o texto
    print("A tokenizar...")
    df_total['tokens'] = df_total['text_clean'].apply(word_tokenize)
    
    #remover stopwords e aplicar lematização
    print("A remover stopwords e a lematizar...")
    df_total['tokens_clean'] = df_total['tokens'].apply(remove_stopwords_and_lemmatize)

    # Juntar os tokens limpos de volta numa string para o TF-IDF
    df_total['final_text'] = df_total['tokens_clean'].apply(lambda x: ' '.join(x))

    # Aplicar TF-IDF
    print("A aplicar TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df_total['final_text'])

    print(X_tfidf.shape)
    
    print("Pré-processamento concluído com sucesso!")
    # Preparar dados para os modelos (Train/Test Split)
    X = X_tfidf
    y = df_total['classe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    main()
