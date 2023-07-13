# NLP: Analizador de Sentimientos

¡Bienvenidos a mi proyecto de análisis de sentimientos! En este proyecto, he puesto en práctica los conocimientos adquiridos en el curso de O'Reilly: Natural Language Processing 2nd Edition, impartido por Bruno Gonçalves.

## Descripción
El objetivo principal de este proyecto es crear un modelo que entienda si el texto de una reseña se corresponde con un texto positivo (es decir, la reseña debería tener 4 estrellas o más) o negativo. Para ello, he entrenado el modelo utilizando datos de reseñas obtenidas de Trustpilot.es.

## Pasos seguidos
A continuación, se detallan los pasos seguidos para llevar a cabo este proyecto:

1. Web Scraping: Utilicé la librería Beautiful Soup de Python para extraer las reseñas de Trustpilot. El tamaño final del dataset utilizado fue de aproximadamente 50.000 filas.

2. Preprocesado de los textos: Implementé técnicas de preprocesado de textos utilizando Regular Expressions y múltiples métodos de la librería "nltk", como "tokenizer" y "stemmers".

3. Entrenamiento de modelos de clasificación: Utilicé las prestaciones ofrecidas por la librería scikit-learn para entrenar varios modelos de clasificación.

## Conclusiones
Como resultado de este proyecto, logré desarrollar un modelo que predice con un 94,74% de acierto si una reseña escrita es positiva (4 estrellas o más) o negativa (3 estrellas o menos). Estoy muy satisfecho con los resultados obtenidos y considero que este proyecto es un gran primer paso en el análisis de sentimientos.

Espero que este proyecto sea de utilidad y pueda inspirar a otros a adentrarse en el apasionante mundo del NLP. ¡Gracias por visitar este repositorio!

¡Un saludo! 😊
