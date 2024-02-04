import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO
import requests
from tmdb_api import fetch_poster
from pathlib import Path

from filters.recommender import Recommender

st.set_page_config(layout="wide")
st.title('Movie Recommender System')
st.subheader('Welcome to the Movie Recommender System built by Shreemit')

def train_test_split_markdown():
    import streamlit as st
    st.markdown(
        """
        ## Train Test Split
        The data is split into training and testing data using the following code:
        ```python
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(ratings, test_size=0.2, random_state=42)
        train.to_csv('train_data.csv', index=False)
        test.to_csv('test_data.csv', index=False)
        ```
        """
    )
with st.sidebar:
    st.title('Movie Recommender System')
    st.subheader('Welcome to the Movie Recommender System built by Shreemit')
    st.write('Please enter a valid User ID from the MovieLens dataset to get movie recommendations.')
    st.write('This app is a movie recommender system that uses the MovieLens dataset to recommend movies to users. The app uses two methods to recommend movies: Content-based filtering and Collaborative filtering. The app also provides documentation on how the system was built and how the data was preprocessed.')

    url = "https://github.com/shreemit/movie-recs"
    st.write("Code can be found here [link](%s)" % url)



# Create tabs
tab1, tab2 = st.tabs(["Recommendations", "Docs"])
with tab1:
 
    # check if userid is numeric or else return error
    with st.container(height = 500):
        # user_id = 1
        user_id = st.text_input('Enter User ID here:', 2)
        if user_id.isnumeric() and int(user_id) > 0:
            user_id = int(user_id)
        else:
            st.write('Please enter a valid User ID')
            st.stop()
        print(user_id)

        # Load the data
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('merge_movies.csv')
        movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))

        # get top-rated movies watched by user
        top5_user_rated = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False)['movieId'].head(5)
        st.write(f"Top 5 movies rated by user {user_id}")
        top5_movies = movies[movies['movieId'].isin(top5_user_rated)]

        # Create a new column 'genres' with space-separated genres
        top5_movies['genres'] = top5_movies['genres'].apply(lambda x: ", ".join(x))

        # Display the data using st.table
        st.table(top5_movies[['title', 'genres']])

        # toggle between content-based and collaborative filtering
        method = st.radio("Choose a recommendation method",
                    ("Content based filtering", "Collaborative filtering"))

    # get top-rated movies watched by user
    recommender = Recommender(ratings_path='train_data.csv', 
                            movies_path='movies.csv', 
                            movie_features_path='movie_features.pkl')


    # get the recommendations
    method = "content_based" if method == "Content based filtering" else "colab_filter"
    movies_to_rec = recommender.recommend_movies(user_id, method=method, k=5)

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

    # st.write(top5_movies_df)

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
            genres[i] = ", ".join(genres[i])
            st.write(
                f'<b style="color:#DB4437">Genres</b>:<b> {genres[i]}</b>',
                unsafe_allow_html=True,
            )
            st.write("________")
with tab2:
    st.write('Documentation')

    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    with st.expander("Introduction"):
        readme = read_markdown_file("README.md")
        st.markdown(readme, unsafe_allow_html=True)
    with st.expander("Train Test Split"):
        train_test = read_markdown_file("docs/train_test_split.md")
        st.markdown(train_test, unsafe_allow_html=True)
    with st.expander("Notebooks"):
        nb_desc = read_markdown_file("docs/notebooks_desc.md")
        st.markdown(nb_desc, unsafe_allow_html=True)
    with st.expander("Model selection"):
        model_sel = read_markdown_file("docs/model_selection.md")
        st.markdown(model_sel, unsafe_allow_html=True)
    with st.expander("Results and Conclusion"):
        eval_conc = read_markdown_file("docs/result.md")
        st.markdown(eval_conc, unsafe_allow_html=True)


