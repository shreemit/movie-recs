import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO
import requests
from tmdb_api import fetch_poster

from filters.colab_filter import create_X, find_similar_movies
from filters.content_based import ContentBasedRecommender

st.set_page_config(layout="wide")
st.title('Movie Recommender System')
st.subheader('Welcome to the Movie Recommender System built by Shreemit')
 
# check if userid is numeric or else return error
user_id = st.text_input('Enter User ID here:')
if user_id.isnumeric() and int(user_id) > 0:
    user_id = int(user_id)
else:
    st.write('Please enter a valid User ID')
    st.stop()
print(user_id)

# Load the data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('merge_movies.csv')


# get top-rated movies watched by user
user_ratings = ratings[ratings['userId'] == user_id]
user_ratings = user_ratings.sort_values('rating', ascending=False)
user_ratings = user_ratings.merge(movies[['movieId', 'title', 'year', 'genres']], on='movieId')
user_top3 = user_ratings.head(3)
st.write(user_top3)

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

# get user top 3 movies



movies_to_rec = set()
    # st.write(f"Top 5 movies similar to {movies[movies['movieId'] == movie]['title'].values[0]}")
    
movie_features = pd.read_pickle('movie_features.pkl')
content_based_recommender = ContentBasedRecommender(movies, movie_features)
user_top3 = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False).head(3)
for movie in user_top3['movieId'].values:
    content_based_movies = content_based_recommender.get_content_based_recommendations(movie_id=movie, 
                                                                                       n_recommendations=3)
    colab_filter_movies = find_similar_movies(movie_id=movie,
                                                movie_mapper=movie_mapper,
                                                movie_inv_mapper=movie_inv_mapper,
                                                X=X,
                                                k=3)
    
    for movie in content_based_movies:
        movies_to_rec.add(movie)
    for movie in colab_filter_movies:
        movies_to_rec.add(movie)

for movie in colab_filter_movies:
    movies_to_rec.add(movie)

# remove the movies that the user has already seen
movies_to_rec = list(movies_to_rec - set(user_ratings['movieId'].values))


# display the recommended movies
st.write(f"Recommended movies for user {user_id}")
top5_movies_df = movies[movies['movieId'].isin(movies_to_rec)][0:5]

titles = []
posters = []
genres = []
for idx, row in top5_movies_df.iterrows():
    titles.append(row['title'])
    posters.append(fetch_poster(row['tmdbId']))
    genres.append(row['genres'])

st.write(top5_movies_df)

placeholder_image = Image.open('placeholder2.png')
cols = [col for col in st.columns(len(titles))]
for i in range(0, len(titles)):
    # print("Index", idx, row)
    with cols[i]:
        st.write(f' <b style="color:#E50914"> {titles[i]} </b>', unsafe_allow_html=True)
        st.write("#")
        if posters[i]:
            st.image(posters[i], use_column_width=True)
        else:
            st.image(placeholder_image, use_column_width=True)
        id = row['movieId']
        st.write("________")
        
        st.write(
            f'<b style="color:#DB4437">Genres</b>:<b> {genres[i]}</b>',
            unsafe_allow_html=True,
        )
