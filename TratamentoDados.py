import re
import os
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

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

# Função para limpar os textos das reviews (ref.https://github.com/dongjun-Lee/transfer-learning-text-tf/blob/master/data_utils.py)
def clean_text(text):
    text = text.lower()                 # Colocar em minúsculas
    text = re.sub(r'[^\w\s]', '', text) # Remover pontuação
    text = re.sub(r'\d+', '', text)     # Remover números
    return text

# (ref.https://github.com/yurayli/imdb_sentiment/blob/master/cnn.py)
def tokenize(train, test, vocab_size=10000, max_length=200):
    # Criar e ajustar o tokenizer no conjunto de treino
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(train)

    # Converter textos para sequências de inteiros
    train_tokens = tokenizer.texts_to_sequences(train)
    test_tokens = tokenizer.texts_to_sequences(test)

    # Aplicar padding para uniformizar o comprimento
    train_padded = pad_sequences(train_tokens, max_length)
    test_padded = pad_sequences(test_tokens, max_length)

    return train_padded, test_padded, tokenizer


#(ref.https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/encoder.py)