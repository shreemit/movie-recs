import numpy as np
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_df, movie_features):
        self.movies_df = movies_df
        self.cosine_sim = cosine_similarity(movie_features, movie_features)
        self.movie_idx = dict(zip(self.movies_dfes['title'], list(self.movies_df.index)))

    def movie_finder(self, title):
        all_titles = self.movies_df['title'].tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    def get_content_based_recommendations(self, title_string, n_recommendations):
        title = self.movie_finder(title_string)
        idx = self.movie_idx[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(n_recommendations+1)]
        similar_movies = [i[0] for i in sim_scores]
        print(f"Recommendations for {title}:")
        print(self.movies_df['title'].iloc[similar_movies])

    

# Usage
# movies = pd.read_csv('movies.csv')    
# cb = ContentBasedRecommender(movies, 'cosine_sim.npy')
# cb.get_content_based_recommendations('The Dark Knight', 10)

