#Embedding: Transforma os índices das palavras em vetores de dimensão = embedding_dim
#GlobalAveragePooling1D: Gera única representação para toda a review
#Dense-relu: Adiciona não-linearidade para o aprendizado de padrões complexos
#Dense-sigmoid: Gera saída entre 0 e 1
#Adam: Otimizador que ajusta os pesos da rede neural para minimizar erros.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

def build_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length),
        GlobalAveragePooling1D(),
        Dense(16, activation = 'relu'),
        Dense(1, activation = 'sigmoid') 
        
    ])
    model.compile(optimizer = Adam(learning_rate = 0.001),
              loss = 'binary_crossentropy', # função de perda para classificação binária
              metrics =['accuracy'])
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    history = model.fit(
        x_train, y_train,  # Dados de treino
        validation_data=(x_val, y_val),  # Dados de teste para validação
        epochs=epochs,  # Número de ciclos
        batch_size=batch_size,  # Tamanho do lote
        verbose=1  # Nível de log
    )
    return history
