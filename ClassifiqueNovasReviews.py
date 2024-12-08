import pandas as pd
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from TratamentoDados import clean_text

def classify_reviews(model, tokenizer, max_length, input_file, output_file):
    
    print("\nClassificação de Novas Resenhas:")
        
    # Ler resenhas do arquivo
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            reviews = file.readlines()
    except FileNotFoundError:
        print("Arquivo não encontrado.")
        return

    # Pré-processar as resenhas
    reviews_cleaned = [clean_text(review) for review in reviews]
    pad_review = tokenizer.texts_to_sequences(reviews_cleaned)
    pad_review = pad_sequences(pad_review, maxlen=max_length)

    # Realizar previsões
    predictions = model.predict(pad_review)
    print("\nResultados da Classificação:")
    
    sentiment = [] 
    
    for i, review in enumerate(reviews):
        #print({"Prediction":predictions[i]})
        if(predictions[i] > 0.7):
            sentiment.append("Positive")
        else:
            sentiment.append("Negative")
        
    result_df = pd.DataFrame({"Review": reviews, "Sentiment": sentiment})
    result_df.to_csv(output_file, index = False)
    print(f"Resultados exportados para o arquivo '{output_file}'")
