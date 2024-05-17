import pandas as pd
import numpy as np
import spacy
from tqdm.notebook import tqdm as tqdm
import faiss
from sentence_transformers import SentenceTransformer

class MovieSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = self.nlp.Defaults.stop_words
        self.index = faiss.read_index('movie_index')
        self.df = pd.read_csv('data_files/movies_preprocessed.csv')

    def preprocess(self, text):
        doc = self.nlp(str(text).lower())
        preprocessed_text = [token.lemma_.lower().strip() for token in doc if not token.is_punct and not token in self.stop_words and not token.is_space]
        return ' '.join(preprocessed_text)

    def search(self, query, k=3):
        preprocessed_query = self.preprocess(query)
        query_embedding = self.model.encode(preprocessed_query)
        distances, indices = self.index.search(np.array(query_embedding).reshape(1, -1), k)
        return self.df.iloc[indices[0]]['title']

if __name__ == "__main__":
    movie_search = MovieSearch()
    results = movie_search.search("good action movies with war in vietnam tom hanks")
    print(results)