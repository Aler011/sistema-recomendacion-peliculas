import json
import os
from flask import Flask, request, jsonify
import numpy as np
from collections import defaultdict
from flask_cors import CORS  # Añade esto al inicio de tus imports

app = Flask(__name__)
CORS(app)  # Añade esta línea después de crear la app

# Archivos JSONz
MOVIES_JSON = "movies.json"
RATINGS_JSON = "ratings.json"

# Datos iniciales
MOVIES = [
    {"Nombre": "Cyber Revolt", "Género": "Sci-Fi", "Descripción": "Un hacker descubre una IA rebelde que controla el mundo."},
    {"Nombre": "Eclipse Carmesí", "Género": "Terror", "Descripción": "Un pueblo olvidado es acechado por una criatura ancestral."},
    {"Nombre": "Más Allá del Cosmos", "Género": "Sci-Fi", "Descripción": "Una misión espacial revela un universo paralelo."},
    {"Nombre": "El Último Cazador", "Género": "Acción", "Descripción": "Un mercenario en busca de venganza en un mundo post-apocalíptico."},
    {"Nombre": "La Ciudad de los Perdidos", "Género": "Drama", "Descripción": "Un detective investiga la desaparición de su hermano."},
    {"Nombre": "Lágrimas de Acero", "Género": "Drama", "Descripción": "Un robot desarrolla emociones humanas en un futuro distópico."},
    {"Nombre": "Destino: Marte", "Género": "Aventura", "Descripción": "Astronautas encuentran señales de vida en Marte."},
    {"Nombre": "Código Letal", "Género": "Thriller", "Descripción": "Un criptógrafo descubre un código que puede cambiar la humanidad."},
    {"Nombre": "Sombras del Pasado", "Género": "Suspenso", "Descripción": "Un periodista desentierra un misterio de la Segunda Guerra Mundial."},
    {"Nombre": "El Eco del Silencio", "Género": "Drama", "Descripción": "Un músico sordo lucha por recuperar su pasión."}
]

INITIAL_RATINGS = [
    {"username": "Ana", "ratings": {"Cyber Revolt": 4, "Eclipse Carmesí": 2, "Destino: Marte": 5, "Lágrimas de Acero": 3, "Código Letal": 4}},
    {"username": "Carlos", "ratings": {"Cyber Revolt": 5, "El Último Cazador": 4, "Código Letal": 3, "Sombras del Pasado": 5, "El Eco del Silencio": 2}},
    {"username": "María", "ratings": {"Eclipse Carmesí": 5, "Más Allá del Cosmos": 3, "La Ciudad de los Perdidos": 4, "Sombras del Pasado": 4, "El Eco del Silencio": 5}},
    {"username": "Pedro", "ratings": {"Cyber Revolt": 2, "El Último Cazador": 5, "Destino: Marte": 4, "Lágrimas de Acero": 1, "Código Letal": 3}},
    {"username": "Laura", "ratings": {"Más Allá del Cosmos": 5, "La Ciudad de los Perdidos": 5, "Destino: Marte": 4, "Sombras del Pasado": 3, "El Eco del Silencio": 4}},
    {"username": "Javier", "ratings": {"Cyber Revolt": 3, "Eclipse Carmesí": 4, "El Último Cazador": 2, "Lágrimas de Acero": 5, "Código Letal": 4}},
    {"username": "Sofía", "ratings": {"Eclipse Carmesí": 1, "Más Allá del Cosmos": 4, "Destino: Marte": 5, "Sombras del Pasado": 2, "El Eco del Silencio": 3}},
    {"username": "Diego", "ratings": {"Cyber Revolt": 5, "El Último Cazador": 5, "La Ciudad de los Perdidos": 3, "Lágrimas de Acero": 4, "Código Letal": 5}},
    {"username": "Elena", "ratings": {"Cyber Revolt": 4, "Más Allá del Cosmos": 3, "Destino: Marte": 2, "Sombras del Pasado": 5, "El Eco del Silencio": 4}},
    {"username": "Pablo", "ratings": {"Eclipse Carmesí": 3, "El Último Cazador": 4, "La Ciudad de los Perdidos": 5, "Lágrimas de Acero": 2, "Código Letal": 1}}
]

def initialize_data():
    """Inicializa los archivos JSON con datos de prueba si no existen"""
    if not os.path.exists(MOVIES_JSON):
        with open(MOVIES_JSON, 'w', encoding='utf-8') as f:
            json.dump(MOVIES, f, ensure_ascii=False, indent=2)
    
    if not os.path.exists(RATINGS_JSON):
        with open(RATINGS_JSON, 'w', encoding='utf-8') as f:
            json.dump(INITIAL_RATINGS, f, ensure_ascii=False, indent=2)

def load_data(filename):
    """Carga datos desde un archivo JSON"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(filename, data):
    """Guarda datos en un archivo JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_rating(username, movie_name, rating):
    """Agrega o actualiza un rating"""
    ratings = load_data(RATINGS_JSON)
    
    # Buscar usuario existente o crear uno nuevo
    user_found = False
    for user_data in ratings:
        if user_data["username"] == username:
            user_data["ratings"][movie_name] = rating
            user_found = True
            break
    
    if not user_found:
        ratings.append({
            "username": username,
            "ratings": {movie_name: rating}
        })
    
    save_data(RATINGS_JSON, ratings)
    return "Rating agregado exitosamente"

def generate_recommendations(username):
    """Genera recomendaciones basadas en similitud de usuarios"""
    movies = load_data(MOVIES_JSON)
    movie_names = [m["Nombre"] for m in movies]
    ratings = load_data(RATINGS_JSON)
    
    # Convertir ratings a formato matriz para cálculos
    user_vectors = {}
    user_ratings_dict = {}  # Para mantener los ratings originales
    
    # Primero creamos un diccionario con todos los usuarios y sus ratings
    for user_data in ratings:
        user_ratings_dict[user_data["username"]] = user_data["ratings"]
    
    # Luego creamos los vectores numéricos para el cálculo de similitud
    for username_key, ratings_dict in user_ratings_dict.items():
        vector = [ratings_dict.get(movie, 0) for movie in movie_names]
        user_vectors[username_key] = vector
    
    if username not in user_vectors:
        return "Usuario no encontrado"
    
    target_vector = user_vectors[username]
    similarities = {}
    
    # Calcular similitud con otros usuarios
    for user, vector in user_vectors.items():
        if user == username:
            continue
        similarity = np.dot(target_vector, vector)
        similarities[user] = similarity
    
    if not similarities:
        return "No hay recomendaciones disponibles"
    
    # Encontrar usuario más similar
    most_similar_user = max(similarities.items(), key=lambda x: x[1])[0]
    
    # Obtener recomendaciones (películas con rating > 3 que el usuario no ha visto)
    recommendations = []
    similar_user_ratings = user_ratings_dict[most_similar_user]
    current_user_ratings = user_ratings_dict[username]
    
    for movie, rating in similar_user_ratings.items():
        if rating > 3 and current_user_ratings.get(movie, 0) == 0:
            recommendations.append(movie)
    
    return recommendations if recommendations else "No hay recomendaciones disponibles"

def register_user(username):
    """Registra un nuevo usuario sin ratings en el sistema"""
    ratings = load_data(RATINGS_JSON)
    
    # Verificar si el usuario ya existe
    for user_data in ratings:
        if user_data["username"] == username:
            return "El usuario ya existe"
    
    # Agregar nuevo usuario con ratings vacíos
    ratings.append({
        "username": username,
        "ratings": {}
    })
    
    save_data(RATINGS_JSON, ratings)
    return "Usuario registrado exitosamente"

# Y el endpoint correspondiente:
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
            
        username = data.get("username")
        
        if not username:
            return jsonify({"error": "Falta el nombre de usuario"}), 400
        
        message = register_user(username)
        return jsonify({"message": message})
        
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

# Endpoints
@app.route('/movies', methods=['GET'])
def get_movies():
    movies = load_data(MOVIES_JSON)
    return jsonify({"movies": movies})

@app.route('/rate', methods=['POST'])
def rate_movie():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
            
        username = data.get("username")
        movie_name = data.get("movie_name")
        rating = data.get("rating")
        
        if not all([username, movie_name, rating]):
            return jsonify({"error": "Faltan datos (username, movie_name o rating)"}), 400
        
        try:
            rating = int(rating)
            if not 1 <= rating <= 5:
                return jsonify({"error": "El rating debe estar entre 1 y 5"}), 400
        except ValueError:
            return jsonify({"error": "Rating debe ser un número"}), 400
        
        movies = load_data(MOVIES_JSON)
        if movie_name not in [m["Nombre"] for m in movies]:
            return jsonify({"error": "Película no encontrada"}), 404
        
        message = add_rating(username, movie_name, rating)
        return jsonify({"message": message})
        
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@app.route('/recommend', methods=['GET'])
def recommend():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "Falta el nombre de usuario"}), 400
    
    recommendations = generate_recommendations(username)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    initialize_data()
    app.run(debug=True, port=5000)