import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#Load data
movies = pd.read_csv('data/movies.csv')  #Make sure to change the path to the file
ratings = pd.read_csv('data/ratings.csv')

#Print the first few rows of the data
print(movies.head())
print(ratings.head())

#Merge the data
movies['combined_features'] = movies['genres'] + " " + movies['title']

#Create a CountVectorizer object
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies['combined_features'])

#Compute the cosine similarity matrix
cosine_sim = cosine_similarity(feature_matrix)

#Function to recommend movies
def recommend_movies(movie_title, num_recommendations=5):
    #Find the index of the movie
    movie_idx = movies[movies['title'] == movie_title].index[0]
    
    #Get the pairwise similarity scores of all movies with that movie
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    
    #Sort the movies based on the similarity scores
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    
    #Add the movies to the list
    recommended_movies = []
    for idx, score in sorted_similar_movies:
        recommended_movies.append(movies.iloc[idx]['title'])
    
    return recommended_movies

#Try the system
movie_to_recommend = "Blue Velvet (1986)"
recommendations = recommend_movies(movie_to_recommend, num_recommendations=5)
print(f"Recommended movies for {movie_to_recommend}: {recommendations}")