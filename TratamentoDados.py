import pandas as pd
import tensorflow as tf
import os
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Caminhos para os dados de treino e teste
train_pos_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'train', 'pos')
train_neg_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'train', 'neg')
teste_pos_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'test', 'pos')
teste_neg_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'test', 'neg')

# Função para carregar todos os arquivos de treino e teste
def load_reviews(folder_path, sentiment):
    data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                review = file.read().strip()
                data.append((review, sentiment))
    return data            
                
data_train = load_reviews(train_pos_path, 1) + load_reviews(train_neg_path, 0)
data_test = load_reviews(teste_pos_path, 1) + load_reviews(teste_neg_path, 0)

# DataFrame
df_train = pd.DataFrame(data_train, columns = ['reviews', 'sentiment'])
df_test = pd.DataFrame(data_test, columns = ['reviews', 'sentiment'])

print(f"Treino: {df_train.shape}, Teste: {df_test.shape}")


# Limpeza de Texto ######################################################################

# Função para limpar os textos das reviews
def clean_text(text):
    text = text.lower() # Colocar em minúsculas
    text = re.sub(r'[^\w\s]', '', text) # Remover pontuação
    text = re.sub(r'\d+', '', text) # Remover números
    return text

# Aplica a função nas reviews do dataframe
df_train['reviews'] = df_train['reviews'].apply(clean_text)
df_test['reviews'] = df_test['reviews'].apply(clean_text)

print(df_train.head())
print(df_test.head())

# Conversão para Tokens ######################################################################
 
# Tokenizar o texto
tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(df_train['reviews'])

# Converter para sequências
train_seq = tokenizer.texts_to_sequences(df_train['reviews'])
test_seq = tokenizer.texts_to_sequences(df_test['reviews'])

# Padronizar o comprimento das sequências (200 palavras por padrão)
train_pad = pad_sequences(train_seq, maxlen = 200)
test_pad = pad_sequences(test_seq, maxlen = 200)

print(f"Formato dos dados de treino: {train_pad.shape}")
print(f"Formato dos dados de teste: {test_pad.shape}")
 