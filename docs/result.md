
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
Our evaluation confirms that collaborative filtering outperforms content-based filtering in recommending movies users enjoy, achieving higher precision, recall, and accuracy. This suggests that user-to-user preference patterns hold stronger predictive power than simply analyzing movie attributes.

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