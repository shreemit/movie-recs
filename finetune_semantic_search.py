import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_data():
    combined_df = pd.read_csv('data_files/combined_df.csv')
    return combined_df

def split_data(combined_df):
    train_queries, test_queries = train_test_split(combined_df, test_size=0.2, random_state=42)
    return train_queries, test_queries

def create_examples(queries):
    examples = []
    for idx, row in tqdm(queries.iterrows()):
        if row['label'] == 1:
            examples.append(InputExample(texts=[row['query_preprocessed'], row['title_preprocessed']]))
        # examples.append(InputExample(texts=[row['query_preprocessed'], row['title_preprocessed']], label=row['label']))
    return examples

def train_model(train_examples, test_examples):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    # train_loss = losses.CosineSimilarityLoss(model)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=1, 
              warmup_steps=100, 
              evaluator=EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='movie-recommender'),
              show_progress_bar=True,
              output_path='../train_models/sebert-movie-search')

def main():
    combined_df = load_data()
    train_queries, test_queries = split_data(combined_df)
    train_examples = create_examples(train_queries)
    test_examples = create_examples(test_queries)
    train_model(train_examples, test_examples)

if __name__ == "__main__":
    main()