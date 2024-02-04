import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO
import requests
from tmdb_api import fetch_poster

from filters.recommender import Recommender

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
top5_user_rated = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False).head(5)
st.write(f"Top 5 movies rated by user {user_id}")

# get top-rated movies watched by user
recommender = Recommender(ratings_path='train_data.csv', 
                          movies_path='movies.csv', 
                          movie_features_path='movie_features.pkl')

# get the recommendations
movies_to_rec = recommender.recommend_movies(user_id, method="content_based", k=5)

# display the recommended movies
st.write(f"Recommended movies for user {user_id}")
top5_movies_df = movies[movies['movieId'].isin(movies_to_rec)][0:5]
# random
top5_movies_df = top5_movies_df.sample(frac=1)

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
