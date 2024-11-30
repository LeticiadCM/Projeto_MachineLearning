import ClassifyNewReviews

model_path = '//caminho//modeloX'
max_length = 200

# Caminho do arquivo de entrada (resenhas) e arquivo de saída (resultados)
input_file = input("\nDigite o caminho do arquivo contendo as resenhas: ")
output_file = f"{input_file}_resultado_classificacao.txt"

# Função de classificação
ClassifyNewReviews.classify_reviews(model_path, tokenizer, max_length, input_file, output_file)
