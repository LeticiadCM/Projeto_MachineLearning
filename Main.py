import ClassifiqueNovasReviews
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

# Caminho para o modelo treinado
model_path = "C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\Projeto_MachineLearning\\ModeloAnaliseDeSentimentos_2024-12-07_21-32-38.h5"
tokenizer_path = "C:\\Users\\letic\\OneDrive\\Documentos\\AprendizadodeMaquina\\Projeto_MachineLearning\\tokenizer_2024-12-07_21-31-28.pkl"
max_length = 200

# Função para carregar o modelo e o tokenizador
def load_tokenizer_and_model(model_path, tokenizer_path):
    
    # Carregar o modelo treinado
    try:
        model = load_model(model_path)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
    
    # Carregar o tokenizador
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print("\nTokenizador carregado.")
    
    return model, tokenizer

# Caminho do arquivo de entrada (resenhas) e arquivo de saída (resultados)
input_file = input("\nDigite o caminho do arquivo contendo as resenhas: ")
output_file = f"{input_file}_resultado_classificacao.txt"

# Função principal
def main():
    try:
        # Carregar modelo e tokenizador
        model, tokenizer = load_tokenizer_and_model(model_path, tokenizer_path)
        
        # Classificar as resenhas do arquivo de entrada
        ClassifiqueNovasReviews.classify_reviews(model, tokenizer, max_length, input_file, output_file)
        
        print(f"\nClassificação concluída. Resultados salvos em: {output_file}")
    
    except Exception as e:
        print(f"\nErro durante a execução: {e}")

if __name__ == "__main__":
    main()
