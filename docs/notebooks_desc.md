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
