from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

#Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

#Configure Flask application
app = Flask(__name__)

#Load data and process features
movies = pd.read_csv('data/movies.csv')
movies['combined_features'] = (movies['genres'] + " ") * 3 + movies['title']
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(feature_matrix)

def normalize_title(title):
    #Remove year from the title and normalize it.
    return title.split('(')[0].strip()

def get_movie_poster(title):
    #Fetch movie poster URL from TMDB API.
    normalized_title = normalize_title(title)
    params = {"api_key": TMDB_API_KEY, "query": normalized_title}
    response = requests.get(f"https://api.themoviedb.org/3/search/movie", params=params)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

#Recommendation function
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

#Main route
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error_message = ""
    searched_movie = ""
    searched_movie_poster = None

    if request.method == "POST":
        movie_title = request.form["movie_title"]
        num_recommendations = request.form.get("num_recommendations")
        
        #Validate number of recommendations
        if not num_recommendations or not num_recommendations.isdigit() or int(num_recommendations) < 1 or int(num_recommendations) > 10:
            error_message = "Please enter a valid number of recommendations (between 1 and 10)."
        else:
            num_recommendations = int(num_recommendations)
            searched_movie = movie_title
            searched_movie_poster = get_movie_poster(movie_title)
            recommendations = recommend_movies(movie_title, num_recommendations=num_recommendations)
            if recommendations[0]["title"] == "Movie not found!":
                error_message = "Movie not found. Please try another title."
                recommendations = []
                searched_movie = ""
                searched_movie_poster = None

    return render_template(
        "index.html",
        recommendations=recommendations,
        error_message=error_message,
        searched_movie=searched_movie,
        searched_movie_poster=searched_movie_poster,
    )


if __name__ == "__main__":
    app.run(debug=True)