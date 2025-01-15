from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Flask application
app = Flask(__name__)

# Load data and process features
movies = pd.read_csv('data/movies.csv')
movies['combined_features'] = movies['genres'] + " " + movies['title']
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(feature_matrix)

# TMDB API configuration
TMDB_API_KEY = "4f9bfdbf145f3000f2ac8ed57a08efe7"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def normalize_title(title):
    """Remove year from the title and normalize it."""
    return title.split('(')[0].strip()

def get_movie_poster(title):
    """Fetch movie poster URL from TMDB API."""
    normalized_title = normalize_title(title)
    params = {"api_key": TMDB_API_KEY, "query": normalized_title}
    response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    try:
        movie_idx = movies[movies['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(cosine_sim[movie_idx]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        recommendations = []
        for idx, _ in sorted_movies:
            title = movies.iloc[idx]['title']
            poster_url = get_movie_poster(title)
            recommendations.append({"title": title, "poster": poster_url})
        return recommendations
    except IndexError:
        return [{"title": "Movie not found!", "poster": None}]

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error_message = ""
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        recommendations = recommend_movies(movie_title)
        if recommendations[0]["title"] == "Movie not found!":
            error_message = "Movie not found. Please try another title."
            recommendations = []
    return render_template("index.html", recommendations=recommendations, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
