# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Reader, Dataset, SVD

# Load the movielens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the dataframes
df = pd.merge(movies, ratings, on='movieId')

# Drop unnecessary columns
df = df.drop(['timestamp'], axis=1)

# Create a user-movie matrix with ratings as values
user_movie_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')

# Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Create a movie-user matrix with ratings as values
movie_user_matrix = user_movie_matrix.T

# Calculate the cosine similarity between movies
movie_similarity = cosine_similarity(movie_user_matrix)

# Convert the similarity matrix to a dataframe
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# Create a tf-idf vectorizer for movie genres
tfidf = TfidfVectorizer(stop_words='english', tokenizer=lambda x: x.split('|'))

# Replace NaN with an empty string
movies['genres'] = movies['genres'].fillna('')

# Construct the tf-idf matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate the cosine similarity between movie genres
genre_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert the similarity matrix to a dataframe
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['movieId'], columns=movies['movieId'])

# Define a function to get the top n similar movies based on user rating and movie genre
def hybrid_recommender(movieId, n):
    # Get the movie title from the movieId
    movie_title = movies[movies['movieId'] == movieId]['title'].iloc[0]
    # Get the similar movies based on user rating
    similar_movies = pd.DataFrame(movie_similarity_df[movieId].sort_values(ascending=False)[1:n+1])
    # Get the similar movies based on movie genre
    similar_genres = pd.DataFrame(genre_similarity_df[movieId].sort_values(ascending=False)[1:n+1])
    # Merge the two dataframes
    hybrid_movies = pd.merge(similar_movies, similar_genres, on='movieId')
    # Calculate the hybrid score by multiplying the user rating similarity and the genre similarity
    hybrid_movies['hybrid_score'] = hybrid_movies['movieId_x'] * hybrid_movies['movieId_y']
    # Sort the movies by hybrid score in descending order
    hybrid_movies = hybrid_movies.sort_values(by='hybrid_score', ascending=False)
    # Get the movie titles from the movieIds
    hybrid_movies['title'] = hybrid_movies['movieId'].apply(lambda x: movies[movies['movieId'] == x]['title'].iloc[0])
    # Return the movie title and the recommended movies
    return movie_title, hybrid_movies['title']

# Define a function to get the top n recommendations for a user based on SVD
def svd_recommender(userId, n):
    # Create a reader object with the rating scale
    reader = Reader(rating_scale=(0.5, 5))
    # Create a dataset object from the ratings dataframe
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    # Build the trainset from the data
    trainset = data.build_full_trainset()
    # Create an SVD object with some parameters
    svd = SVD(n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, verbose=True)
    # Train the SVD model on the trainset
    svd.fit(trainset)
    # Get the list of all movieIds
    movieIds = movies['movieId'].unique()
    # Get the list of movieIds that the user has rated
    rated_movieIds = ratings[ratings['userId'] == userId]['movieId'].unique()
    # Get the list of movieIds that the user has not rated
    unrated_movieIds = [x for x in movieIds if x not in rated_movieIds]
    # Get the predicted ratings for the unrated movies
    predicted_ratings = {}
    for movieId in unrated_movieIds:
        predicted_ratings[movieId] = svd.predict(userId, movieId).est
    # Sort the predicted ratings in descending order
    predicted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    # Get the top n movieIds
    top_movieIds = [x[0] for x in predicted_ratings[:n]]
    # Get the movie titles from the movieIds
    top_movies = movies[movies['movieId'].isin(top_movieIds)]['title']
    # Return the recommended movies
    return top_movies

# Test the hybrid recommender function
movieId = 1 # Toy Story (1995)
n = 10 # Number of recommendations
movie_title, hybrid_movies = hybrid_recommender(movieId, n)
print(f'For the movie {movie_title}, the top {n} hybrid recommendations are:')
print(hybrid_movies)

# Test the svd recommender function
userId = 1 # User ID
n = 10 # Number of recommendations
svd_movies = svd_recommender(userId, n)
print(f'For the user {userId}, the top {n} SVD recommendations are:')
print(svd_movies)
