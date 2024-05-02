# Movie Recommender System

This repository contains the code for a movie recommender system. The system uses both content-based and collaborative filtering methods to recommend movies to users.

## Files in this repository

### Python Scripts

- `app.py`: This is the main script for the movie recommender system. It uses Streamlit to create a web-based user interface for the recommender system.

- `content_based_recommender.py`: This script contains the `ContentBasedRecommender` class, which implements the content-based filtering method.

- `collaborative_filter.py`: This script contains the `CollaborativeMovieRecommender` class, which implements the collaborative filtering method.

### Jupyter Notebooks

- `data_preprocessing.ipynb`: This notebook contains the code for preprocessing the movie and ratings data.

- `model_training.ipynb`: This notebook contains the code for training the recommender system.

- `model_evaluation.ipynb`: This notebook contains the code for evaluating the performance of the recommender system.

## How to run the recommender system

1. Clone this repository to your local machine.
2. Install the required Python packages using pip: `pip install -r requirements.txt`
3. Run the streamlit `app.py` script: `streamlit run app.py`
4. Open your web browser and navigate to `http://localhost:8501` to use the recommender system.
5. You can also access the web app [here](https://movierecs.streamlit.app/)

## Model Selection Rational
The recommendation system in this repository uses both collaborative filtering and content-based filtering methods to recommend movies. 

The rationale for using collaborative filtering is that it leverages the wisdom of the crowd. It assumes that if users agreed in the past, they will agree in the future. This method is effective when there is a large amount of user interaction data available.

On the other hand, the content-based filtering method was chosen because it recommends items by comparing the content of the items to a user profile. The content of each item is represented as a set of descriptors, such as the words in a document. This method is particularly useful when there is detailed information about each item, but not much information about the users.

The model was tested using precision, recall, and accuracy metrics. The results showed that the model was able to effectively recommend movies that users rated highly in the test data. 

Based on the testing and evaluation, it was found that the collaborative filtering method outperformed the content-based filtering method. This could be due to the fact that collaborative filtering leverages the collective preferences of users, which can often provide more accurate recommendations than relying solely on the content of the items.

## Notebooks
Notebooks in this repository are used for data preprocessing, model training, and model evaluation. The following is a brief description of each notebook:
- `data_preproc.ipynb`
- `linking.ipynb`
- `collab_filter.ipynb`
- `content_based.ipynb`
- `evaluation.ipynb`

### Notebook: `data_preproc.ipynb`
Initial data preprocessing is performed in this notebook.
#### Key Steps:
1. **Data Loading & Cleaning**: Load data from CSV files, check for missing values, and remove duplicates.
2. **Data Transformation**: Extract year from movie title, clean title text, split genres into a list, and convert year to numeric.
3. **Data Analysis**: Calculate user rating frequency and mean movie rating, identify highest and lowest rated movies.
4. **Data Merging**: Merge `movies` and `imdb_movies` DataFrames on movie title, and check for missing values.

### Notebook: `linking.ipynb`

This notebook integrates data from multiple sources (links, movies, IMDb, TMDb) to create a comprehensive movie dataset with consistent formatting and TMDb IDs.
#### Key steps:
1. Load and prepare data:

    - Imports necessary libraries (pandas, re).
    - Loads data from CSV files.
    - Renames columns for consistency.

2. Merges datasets:
    - Combine given movie data with link data and IMDb data.
    - Extract year and cleans movie titles.
    - Merge TMDb data with the main movie data.
3. Handle missing data:
    - Identifies and addresses missing values.
4. Save the merged data.
    - Creates a merged CSV file with TMDb IDs (merge_movies.csv).

The main utility of adding TMDb IDs is that they can be used with the TMDb API to fetch additional movie data.

### Notebook: `collab_filter.ipynb`
This notebook contains the code for training a collaborative filtering-based movie recommender system.

#### Key Steps:
2. **Data Exploration**: Calculate and print statistics about the data, such as the number of ratings, unique movies, and unique users.

3. **Data Visualization**: Plot the distribution of movie ratings and the number of movies rated per user.

4. **Rating Analysis**: Calculate the mean rating for each movie, identify the highest and lowest rated movies, and calculate the Bayesian average rating for each movie.

6. **Collaborative Filtering with k-Nearest Neighbors**: Implement a function to create a user-item utility matrix, a function to find similar movies for a given movie, and a function to recommend movies for a given user.

7. **Movie Recommendation**: Use the implemented functions to recommend movies for a user based on the movies they have rated highly.

### Notebook: `content_based.ipynb`
This notebook contains the code for training a content-based movie recommender system.
#### Key Steps:
3. **Data Visualization**: Plot the distribution of movie genres and the number of movies per decade.

4. **Data Transformation**: Extract the year from the movie title, round down the year to the nearest decade, and create binary features for each genre and decade.

5. **Feature Extraction**: Concatenate the genre and decade features to create a feature matrix for each movie.

6. **Similarity Calculation**: Calculate the cosine similarity between each pair of movies based on their feature matrices.

7. **Movie Finder**: Implement a function to find the movie in the data that most closely matches a given title.

8. **Recommendation Function**: Implement a function to recommend movies that are most similar to a given movie.

### Notebook: `evaluation.ipynb`
This Jupyter notebook contains the code for evaluating the performance of the movie recommender system.

#### Key Steps:
3. **Initialize Recommender**: Initialize the Recommender class with paths to the training data, movie details, and movie features.

4. **Recommend Movies**: Use the Recommender class to recommend movies for a specific user using the content-based method.

5. **Loop Through Test Data**: Loop through the test data and use the Recommender class to recommend movies for each user using both the collaborative filtering and content-based methods.

6. **Prepare Test Data**: Prepare a dictionary of movies that each user in the test data has rated 3 or higher.

7. **Evaluate Recommendations**: Evaluate the recommendations made by the system using precision, recall, and accuracy metrics.


## Evaluation, Results and Discussion
The table below compares the performance of two movie recommendation methods: collaborative filtering and content-based filtering. 

To evaluate these methods, we first identified movies that each user in our test data rated 3 or higher. This served as our "ground truth". We then used both methods to recommend movies to each user. 

We used three metrics to evaluate the quality of these recommendations: precision, recall, and accuracy. For each recommended movie, we checked if it was in the user's ground truth. If it was, we marked it as a successful recommendation (1), otherwise, it was marked as unsuccessful (0). We limited our evaluation to the top 5 movie recommendations for each user.


| Model                 | Precision | Recall | Accuracy |
|-----------------------|-----------|--------|----------|
| Collaborative Filtering |    0.39   |  0.41  |   0.39   |
| Content-Based Filtering |    0.08   |  0.08  |   0.08   |


The table shows that the collaborative filtering method was more successful at recommending movies that users liked, as indicated by its higher precision, recall, and accuracy scores.

## Conclusion and Future Work with Insights:
Our evaluation confirms that collaborative filtering outperforms content-based filtering in recommending movies users enjoy, achieving higher precision, recall, and accuracy. 

While both the collaborative filtering and content-based filtering methods have their strengths, neither method has shown exceptional performance in our tests. There is significant room for improvement in both accuracy and personalization of the recommendations. This highlights the need for exploring more advanced techniques, such as transformer models and hybrid models, to enhance the effectiveness of our movie recommendation system.

### Looking Ahead:

While this is a promising result, here are exciting avenues for exploration:

1. Enhancing Dataset Size:

    To enhance our recommendation models, we can integrate more detailed information about movies. This could involve sourcing actor details from databases like IMDb, utilizing natural language processing to derive plot summaries from publicly accessible resources, and incorporating additional movie attributes such as release year, director, genre, ratings, and awards. These improvements could provide a more comprehensive understanding of each movie, potentially leading to more accurate and personalized recommendations.

1. Dive Deeper with Transformer Models:

    Leverage the impressive learning capabilities of transformer models like BERT2Rec. These models excel at capturing complex relationships in data, potentially refining recommendation quality.
    Analyze how well BERT2Rec handles cold-start issues (lack of ratings for new users/movies) compared to our current approach.
2. Harness a Hybrid Powerhouse:

    Create a hybrid model that merges the strengths of both collaborative and content-based filtering using a neural network.
    Investigate how effectively this hybrid approach personalizes recommendations by considering both user preferences and movie content.
    Compare the performance of the hybrid model against individual baselines and the current best model.




##### These improvements could be implemented to potentially improve the recommendation system's accuracy and personalization capabilities.

### References:
1. [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
2. [GitHub Tutorial]('https://github.com/topspinj')
3. [Streamlit App]('https://github.com/vikramr22/moviesway-v2')
4. ChatGPT, Bard

In the `notebooks/splitting.ipynb` notebook, the code first creates a DataFrame `user_ids_df` that groups the ratings by "userId" and counts the number of ratings for each user. This DataFrame provides an overview of the distribution of ratings per user.

To ensure a fair and robust evaluation of the recommender system, the data is split into training and test sets in a way that each user's ratings are equally divided. This is crucial because it ensures that the model has information about every user in both the training and test sets.

Here's how this is achieved:

```python
from sklearn.model_selection import train_test_split

# Empty dataframes for training and test sets are created
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# For each unique user, their ratings are split into training and test sets
for user_id in ratings['userId'].unique():
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # The user's ratings are split into training and test sets
    user_train, user_test = train_test_split(user_ratings, test_size=0.5, random_state=42)
    
    # The user's training and test sets are appended to the overall training and test sets
    train_data = train_data.append(user_train)
    test_data = test_data.append(user_test)
```

In this code, the `train_test_split` function is used to split each user's ratings into a training set (50% of their ratings) and a test set (50% of their ratings). The `random_state` parameter is set to ensure that the split is reproducible. The training and test sets for each user are then appended to the overall training and test sets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.