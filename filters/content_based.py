import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_df, movie_features):
        self.movies_df = movies_df
        self.cosine_sim = cosine_similarity(movie_features, movie_features)
        self.movie_idx_mapper = dict(zip(movies_df['movieId'], list(movies_df.index)))
    
    def get_content_based_recommendations(self, movie_id, n_recommendations):
        idx = self.movie_idx_mapper[movie_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(n_recommendations+1)]
        similar_movies = [i[0] for i in sim_scores]
        # print(f"Recommendations for {title}:")
        return similar_movies

    

# Usage
# movies = pd.read_csv('movies.csv')    
# cb = ContentBasedRecommender(movies, 'cosine_sim.npy')
# cb.get_content_based_recommendations('The Dark Knight', 10)

