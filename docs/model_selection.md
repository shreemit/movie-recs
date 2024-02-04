## Model Selection Rational
The recommendation system in this repository uses both collaborative filtering and content-based filtering methods to recommend movies. 

The rationale for using collaborative filtering is that it leverages the wisdom of the crowd. It assumes that if users agreed in the past, they will agree in the future. This method is effective when there is a large amount of user interaction data available.

On the other hand, the content-based filtering method was chosen because it recommends items by comparing the content of the items to a user profile. The content of each item is represented as a set of descriptors, such as the words in a document. This method is particularly useful when there is detailed information about each item, but not much information about the users.

The model was tested using precision, recall, and accuracy metrics. The results showed that the model was able to effectively recommend movies that users rated highly in the test data. 

Based on the testing and evaluation, it was found that the collaborative filtering method outperformed the content-based filtering method. This could be due to the fact that collaborative filtering leverages the collective preferences of users, which can often provide more accurate recommendations than relying solely on the content of the items.
