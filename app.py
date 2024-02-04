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
if user_id.isnumeric():
    user_id = int(user_id)
else:
    st.write('Please enter a valid User ID')
print(user_id)

# Load the data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('merge_movies.csv')

# get top-rated movies watched by user
user_ratings = ratings[ratings['userId'] == user_id]
user_ratings = user_ratings.sort_values('rating', ascending=False)
user_ratings = user_ratings.merge(movies[['movieId', 'title', 'year']], on='movieId')

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

# get user top 3 movies
user_top3 = user_ratings.head(3)
st.write(user_top3)

movie_to_rec = set()
# get top 5 similar movies for top 3 movies
for movie in user_top3['movieId']:
    # st.write(f"Top 5 movies similar to {movies[movies['movieId'] == movie]['title'].values[0]}")
    similar_movies = find_similar_movies(movie, movie_mapper, movie_inv_mapper, X, k=5)
    movie_to_rec.update(similar_movies[0:2])


# display the recommended movies
st.write(f"Recommended movies for user {user_id}")
top5_movies_df = movies[movies['movieId'].isin(movie_to_rec)][0:5]

titles = []
posters = []
genres = []
for idx, row in top5_movies_df.iterrows():
    titles.append(row['title'])
    posters.append(fetch_poster(row['movieId']))
    genres.append(row['genres'])

st.write(top5_movies_df)

placeholder_image = Image.open('placeholder2.png')
cols = [col for col in st.columns(len(titles))]
st.write(len(cols), len(titles))
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
            f'<b style="color:#DB4437">Rating</b>:<b> {genres[i]}</b>',
            unsafe_allow_html=True,
        )
