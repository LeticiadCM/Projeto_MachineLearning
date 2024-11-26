import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Função para limpar os textos das reviews
def clean_text(text):
    text = text.lower() # Colocar em minúsculas
    text = re.sub(r'[^\w\s]', '', text) # Remover pontuação
    text = re.sub(r'\d+', '', text) # Remover números
    return text

def tokenize_pad(texts, vocab_size = 10000, max_length = 200):
    tokenizer = Tokenizer(num_words = 5000, oov = '<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    pad_sequences = pad_sequences(sequences, maxlen = max_length)
    return pad_sequences, tokenizer
