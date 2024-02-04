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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.