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