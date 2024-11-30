import TratamentoDados
import pandas as pd
import os
import datetime
#import seaborn as sns
#import matplotlib.pyplot as plt
from TratamentoDados import load_reviews, clean_text, tokenize
from TreinamentoModelo import build_model, train_model
from sklearn.metrics import classification_report, confusion_matrix

vocab_size = 10000  # Tamanho do vocabulário
embedding_dim = 64  # Dimensão dos vetores de embedding
max_length = 200    # Comprimento máximo das sequências
batch_size = 32     # Tamanho do lote
epochs = 5          # Número máximo de épocas

# Caminhos para os dados de treino e teste
train_pos_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'train', 'pos')
train_neg_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'train', 'neg')
teste_pos_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'test', 'pos')
teste_neg_path = os.path.join("C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\aclImdb_v1\\", 'test', 'neg')            

print("Lendo dados...")                
data_train = load_reviews(train_pos_path, 1) + load_reviews(train_neg_path, 0)
data_test = load_reviews(teste_pos_path, 1) + load_reviews(teste_neg_path, 0)

# DataFrame
df_train = pd.DataFrame(data_train, columns = ['reviews', 'sentiment'])
df_test = pd.DataFrame(data_test, columns = ['reviews', 'sentiment'])

#df_train = df_train.sample(5000)  # Amostra de 5000 reviews
#df_test = df_test.sample(5000)   # Amostra de 5000 reviews

print(f"Treino: {df_train.shape}, Teste: {df_test.shape}")

# Pré-processamento
print("Limpando dados...")   
df_train['reviews'] = df_train['reviews'].apply(clean_text)
df_test['reviews'] = df_test['reviews'].apply(clean_text)

# Tokenizer e padding
print("Tokenizando e aplicando padding...")
pad_train, pad_test, tokenizer = tokenize(df_train['reviews'], df_test['reviews'], vocab_size, max_length)

labels_train = df_train['sentiment'].values # variáveis dependentes
labels_test = df_test['sentiment'].values

print("Treinando o modelo...")
model = build_model(vocab_size, embedding_dim, max_length)
history = train_model(model, pad_train, labels_train, pad_test, labels_test, epochs, batch_size)

# Salvar o modelo treinado
model.save(f"ModeloAnaliseDeSentimentos{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5")

print(f"Formato dos dados de treino: {pad_train.shape}")
print(f"Formato dos dados de teste: {pad_test.shape}")

results = model.evaluate(pad_test, labels_test, verbose=1)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")


 