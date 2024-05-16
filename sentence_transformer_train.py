from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import Adam
from torch.nn import CosineEmbeddingLoss

class TextSimilarityModel:
    def __init__(self, corpus_name, rel_name, model_name='all-MiniLM-L6-v2', top_k=10):
        """
        Initialize the model with datasets and pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.corpus_name = corpus_name
        self.rel_name = rel_name
        self.top_k = top_k
        self.load_data()

    def load_data(self):
        """
        Load and filter datasets based on test queries and documents.
        """
        # Load query and document datasets
        dataset_queries = load_dataset(self.corpus_name, "queries")
        dataset_docs = load_dataset(self.corpus_name, "corpus")

        # Extract queries and documents
        self.queries = dataset_queries["queries"]["text"]
        self.query_ids = dataset_queries["queries"]["_id"]
        self.documents = dataset_docs["corpus"]["text"]
        self.document_ids = dataset_docs["corpus"]["_id"]

        # Buil query and document id to text mapping
        self.query_id_to_text = {query_id:query for query_id, query in zip(self.query_ids, self.queries)}
        self.document_id_to_text = {document_id:document for document_id, document in zip(self.document_ids, self.documents)}

        # Build relevant queries and documents mapping based on train set 
        train_qrels = load_dataset(self.rel_name)["train"] #TODO
        self.train_query_id_to_relevant_doc_ids = {qid: [] for qid in train_qrels["query-id"]}
        for qid, doc_id in zip(train_qrels["query-id"], train_qrels["corpus-id"]):
            if qid in self.train_query_id_to_relevant_doc_ids:
                self.train_query_id_to_relevant_doc_ids[qid].append(doc_id)

        # Filter queries and documents and build relevant queries and documents mapping based on test set
        test_qrels = load_dataset(self.rel_name)["test"]
        self.filtered_test_query_ids = set(test_qrels["query-id"])
        self.filtered_test_doc_ids = set(test_qrels["corpus-id"])

        self.test_queries = [q for qid, q in zip(self.query_ids, self.queries) if qid in self.filtered_test_query_ids]
        self.test_query_ids = [qid for qid in self.query_ids if qid in self.filtered_test_query_ids]
        self.test_documents = [doc for did, doc in zip(self.document_ids, self.documents) if did in self.filtered_test_doc_ids]
        self.test_document_ids = [did for did in self.document_ids if did in self.filtered_test_doc_ids]

        
        self.test_query_id_to_relevant_doc_ids = {qid: [] for qid in self.test_query_ids}
        for qid, doc_id in zip(test_qrels["query-id"], test_qrels["corpus-id"]):
            if qid in self.test_query_id_to_relevant_doc_ids:
                self.test_query_id_to_relevant_doc_ids[qid].append(doc_id)

    def rank_documents(self):
        """
        """
        #TODO Part 1:
        ###########################################################################
        query_embeddings = self.model.encode(self.test_queries)
        document_embeddings = self.model.encode(self.test_documents)

        self.query_id_to_ranked_doc_ids = {}
        for qid, query_embedding in zip(self.test_query_ids, query_embeddings):
            similarities = cosine_similarity([query_embedding], document_embeddings)[0]
            ranked_doc_indices = np.argsort(similarities)[::-1][:self.top_k]
            self.query_id_to_ranked_doc_ids[qid] = [self.test_document_ids[idx] for idx in ranked_doc_indices]
        ###########################################################################

    @staticmethod
    def average_precision(relevant_docs, candidate_docs):
        """
        Compute average precision for a single query.
        """
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
        precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k]]
        return np.mean(precisions) if precisions else 0

    def mean_average_precision(self):
        """
        Compute mean average precision for all queries.
        """
        ap_scores = [self.average_precision(self.test_query_id_to_relevant_doc_ids[qid], 
                                            self.query_id_to_ranked_doc_ids.get(qid, []))
                     for qid in self.test_query_id_to_relevant_doc_ids]
        return np.mean(ap_scores)

    def fine_tune_model(self, batch_size=16, num_epochs=3, save_model_path="finetuned_senBERT"):
        """
        Fine-tunes the model using MultipleNegativesRankingLoss.
        """
        # Load training data
        train_mapping_ids = load_dataset(self.rel_name, "default")["train"] #TODO 
        
        # Prepare training examples
        train_examples = self.prepare_training_examples(train_mapping_ids)

        # Create DataLoader for training data
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        # Define the MultipleNegativesRankingLoss
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        # Start fine-tuning
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], 
                       epochs=num_epochs, 
                       warmup_steps=100, 
                       show_progress_bar=True, 
                       output_path=save_model_path)


    def prepare_training_examples(self, train_mapping_ids):
        """
        Prepares training examples from the training data.
        """
        train_examples = []
        used_doc_ids = []
        for qid, doc_ids in self.train_query_id_to_relevant_doc_ids.items():
            usable_doc_ids = set(doc_ids) - set(used_doc_ids)
            for doc_id in usable_doc_ids:
                anchor = self.query_id_to_text[qid]
                positive = self.document_id_to_text[doc_id]
                train_examples.append(InputExample(texts=[anchor, positive]))
            used_doc_ids.extend(doc_ids)

        return train_examples


# Initialize the model  

#model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", model_name='finetuned_senBERT_train')
# model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", model_name='all-MiniLM-L6-v2')
# model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", model_name='Snowflake/snowflake-arctic-embed-xs')

# Finetune the model  
model.fine_tune_model2(batch_size=4, num_epochs=1, save_model_path="finetuned_senBERT_testset_v4")  # Adjust batch size and epochs as needed

model.rank_documents()
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


# train set training 44.96 MAP
# test set training 47.xx MAP