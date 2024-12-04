# Embedding: Transforma os índices das palavras em vetores de dimensão = embedding_dim
# GlobalAveragePooling1D: Gera única representação para toda a review
# Dense-relu: Adiciona não-linearidade para o aprendizado de padrões complexos
# Dense-sigmoid: Gera saída entre 0 e 1
# Adam: Otimizador que ajusta os pesos da rede neural para minimizar erros.

from keras.models import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Construção do modelo (ref.https://github.com/dongjun-Lee/transfer-learning-text-tf/blob/master/train.py)
def build_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length),
        GlobalAveragePooling1D(),
        Dropout(0.25),
        Dense(16, activation = 'relu'),
        Dense(1, activation = 'sigmoid') 
        
    ])
    model.compile(optimizer = Adam(learning_rate = 0.001),
              loss = 'binary_crossentropy', # Função de perda para classificação binária
              metrics =['accuracy'])
    
    return model

# Configurando o EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',     # Métrica a ser monitorada
    patience=2,                 # Número de épocas sem melhora antes de parar
    restore_best_weights=True   # Restaurar os melhores pesos após parada
)

# Treinamento do modelo (ref.https://github.com/raminmh/CfC/blob/main/train_imdb.py)
def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    history = model.fit(
        x_train, y_train,                # Dados de treino
        validation_data=(x_val, y_val),  # Dados de teste para validação
        epochs=epochs,                   # Número de épocas
        batch_size=batch_size,           # Tamanho do lote
        verbose=1,                       # Nível de log
        callbacks = [early_stopping]
    )
    return history
