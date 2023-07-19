from flask import Flask, render_template, request
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
import pandas as pd
from sklearn.model_selection import train_test_split


stopwords = nltk.corpus.stopwords.words('spanish')
stemmer = SnowballStemmer("spanish")
def preprocess_text(text):
    # Eliminar puntuación
    text = ''.join([c for c in text if c not in punctuation])

    # Convertir a minúsculas
    text = text.lower()

    # Tokenización
    tokens = nltk.word_tokenize(text)

    # # Eliminar palabras de parada
    # tokens = [word for word in tokens if word not in stopwords]

    #Aplicar un stemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text



app = Flask(__name__)

# Cargamos el modelo
with open('model/nlp_model_95%.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)

# Cargamos el vectorizador
with open('model/vectorizador.pkl', 'rb') as archivo:
    vectorizador = pickle.load(archivo)

# Ruta de inicio
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el texto ingresado en el formulario
    texto = request.form['texto']

    processed_text = preprocess_text(texto)
    new_text_vector = vectorizador.transform([processed_text])
    prediction = modelo.predict(new_text_vector)

    if prediction == [0]:
        respuesta = 'Es una mala reseña, menos de 3 estrellas fijo'
    else:
        respuesta = 'Es una buena reseña, más de 4 estrellas seguro'


    # Mostrar la predicción en la página de resultados
    return render_template('resultados.html', prediccion=respuesta, texto=texto)

if __name__ == '__main__':
    app.run(debug=True)
