import sys

sys.path.insert(0, "/Users/shreemit/Developer/movie-recs")
from filters.content_based import ContentBasedRecommender
from filters.colab_filter import CollaborativeMovieRecommender
import pandas as pd


class Recommender:
    def __init__(self, ratings_path, movies_path, movie_features_path):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.movie_features_path = movie_features_path
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.movie_features = pd.read_pickle(movie_features_path)

    def recommend_movies(self, user_id, method="colab_filter", k=5):
        # get top-rated movies watched by user
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        user_ratings = user_ratings.sort_values("rating", ascending=False)
        user_top3 = user_ratings.head(3)

        movies_to_rec = []

        if method == "content_based":
            content_based_recommender = ContentBasedRecommender(self.movies, self.movie_features)
            content_based_recs = []
            for movieId in user_top3["movieId"].values:
                content_based_rec = (
                    content_based_recommender.get_content_based_recommendations(
                        movieId, 5
                    )
                )
                content_based_rec = {
                    index + 1: movie for index, movie in enumerate(content_based_rec)
                }
                content_based_recs.append(content_based_rec)
            movies_to_rec = content_based_recs
        elif method == "colab_filter":
            colab_filter = CollaborativeMovieRecommender(self.movies, self.ratings)
            colab_filter.preprocess_data()
            colab_filter.create_X()

            # find similar movies for the top-rated movies watched by the user
            colab_filter_movies = []
            for movieId in user_top3["movieId"].values:
                colab_filter_movie = colab_filter.find_similar_movies(
                    movie_id=movieId, k=5
                )
                colab_filter_movie = {
                    index + 1: movie for index, movie in enumerate(colab_filter_movie)
                }
                colab_filter_movies.append(colab_filter_movie)
            movies_to_rec = colab_filter_movies

        keys = sorted(
            set(key for d in movies_to_rec for key in d), key=lambda x: (x != 1, x)
        )

        complete_recs = []
        for key in keys:
            for item in movies_to_rec:
                value = item.get(key, None)
                if value is not None:
                    complete_recs.append(value)

        # remove the movies that the user has already seen
        complete_recs = [
            movie for movie in complete_recs if movie not in user_top3["title"].values
        ]
        return complete_recs[:k]
