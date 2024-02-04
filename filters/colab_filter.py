import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.sparse import save_npz

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the data
movies = pd.read_csv('merge_movies.csv')
ratings = pd.read_csv('ratings.csv')    

# Merge the data
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')
print(movie_stats.columns, movies.columns, bayesian_avg_ratings.columns)

# Merge the data
movie_stats = movie_stats.merge(movies['movieId'])
movie_stats = movie_stats.sort_values('bayesian_avg', ascending=False)

# Collaborative Filtering
def create_X(df):
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


def find_similar_movies(movie_id, movie_mapper, movie_inv_mapper, X, k, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Returns:
        list of k similar movie ID's
    """
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

def find_recommended_movies(user_id, user_mapper, movie_inv_mapper, X, k, metric='cosine', show_distance=False):
    """
    Finds k-recommended movies for a given user id.
    
    Args:
        user_id: id of the user of interest
        X: user-item utility matrix
        k: number of recommended movies to retrieve
        metric: distance metric for kNN calculations
    
    Returns:
        list of k recommended movie ID's
    """
    recommended_ids = []
    
    user_ind = user_mapper[user_id]
    user_vec = X[:, user_ind]
    # Find the movies that the user has rated 4 or higher
    high_rated_movies = np.where(user_vec >= 4)[0]
    # For each high rated movie, find similar movies using kNN
    for movie_ind in high_rated_movies:
        movie_id = movie_inv_mapper[movie_ind]
        movie_vec = X[movie_ind]
        kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
        kNN.fit(X)
        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1,-1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
        for i in range(1,k+1):
            n = neighbour.item(i)
            recommended_ids.append(movie_inv_mapper[n])
    # Remove duplicates and movies that the user has already seen
    recommended_ids = list(set(recommended_ids) - set(high_rated_movies))
    # Return the first k movies
    return recommended_ids[:k]

def recommend_movies_for_user(user_id, user_mapper,, movie_inv_mapper, X, k, metric='cosine'):
    """
    Recommends k movies for a given user_id.

    Args:
        user_id: id of the user for whom to make recommendations
        user_mapper: dictionary mapping user_ids to indices in the utility matrix
        movie_inv_mapper: dictionary mapping indices in the utility matrix to movie_ids
        X: user-item utility matrix
        k: number of movies to recommend
        metric: distance metric for kNN calculations

    Returns:
        list of k recommended movie ID's
    """

    user_ind = user_mapper[user_id]
    user_vec = X[:, user_ind]  # Get the user's vector of ratings

    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    knn.fit(X.T)  # Fit on transposed X to find similar users

    distances, indices = knn.kneighbors(user_vec.reshape(1, -1))
    neighbor_user_ids = [user_inv_mapper[idx] for idx in indices.flatten()[1:]]  # Exclude first result (user itself)

    recommended_movie_ids = []
    for neighbor_user_id in neighbor_user_ids:
        user_ratings = ratings[ratings['userId'] == neighbor_user_id]
        rated_movie_ids = user_ratings['movieId'].tolist()
        for movie_id in rated_movie_ids:
            if movie_id not in recommended_movie_ids and movie_id not in ratings[ratings['userId'] == user_id]['movieId'].tolist():
                recommended_movie_ids.append(movie_id)
                if len(recommended_movie_ids) == k:
                    break  # Stop if we found k recommended movies

    return recommended_movie_ids