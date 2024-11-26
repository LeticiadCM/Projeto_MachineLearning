import pandas as pd
import os

from TratamentoDados import load_reviews, clean_text, tokenize_pad
from TreinamentoModelo import build_model, train_model

vocab_size = 10000  # Tamanho do vocabulário
embedding_dim = 16  # Dimensão dos vetores de embedding
max_length = 200    # Comprimento máximo das sequências

# Caminhos para os dados de treino e teste
train_pos_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'train', 'pos')
train_neg_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'train', 'neg')
teste_pos_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'test', 'pos')
teste_neg_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'test', 'neg')            
                
data_train = load_reviews(train_pos_path, 1) + load_reviews(train_neg_path, 0)
data_test = load_reviews(teste_pos_path, 1) + load_reviews(teste_neg_path, 0)

# DataFrame
df_train = pd.DataFrame(data_train, columns = ['reviews', 'sentiment'])
df_test = pd.DataFrame(data_test, columns = ['reviews', 'sentiment'])

print(f"Treino: {df_train.shape}, Teste: {df_test.shape}")

# Pré-processamento
df_train['reviews'] = df_train['reviews'].apply(clean_text)
df_test['reviews'] = df_test['reviews'].apply(clean_text)

print(df_train.head())
print(df_test.head())

pad_train, tokenizer = tokenize_pad(df_train['reviews'])
pad_test, _ = tokenize_pad(df_test['review'], tokenizer)

labels_train = df_train['sentiment'].values # variáveis dependentes
labels_test = df_test['sentiment'].values

model = build_model()

history = train_model(model, pad_train, labels_train, pad_test, labels_test)

print(f"Formato dos dados de treino: {pad_train.shape}")
print(f"Formato dos dados de teste: {pad_test.shape}")
 