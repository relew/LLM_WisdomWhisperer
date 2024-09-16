import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Index:
    """
    A search index using TF-IDF for text fields, exact matching for keyword fields,
    and cosine similarity for vector fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        vector_fields (list): List of vector field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        vector_matrices (dict): Dictionary of vector matrices for each vector field.
        docs (list): List of documents indexed.
    """

    def __init__(self, text_fields, keyword_fields, vector_fields=None, vectorizer_params={}):
        """
        Initializes the Index with specified text, keyword, and vector fields.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vector_fields (list): List of vector field names to index (optional).
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.vector_fields = vector_fields or []
        
        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.vector_matrices = {}
        self.docs = []

    def fit(self, docs):
        """
        Fits the index with the provided documents.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)

        for field in self.vector_fields:
            vectors = [doc.get(field, np.zeros(0)) for doc in docs]
            self.vector_matrices[field] = np.vstack(vectors)

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10, vector_query=None):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.
            vector_query (np.array): Optional query vector for vector field search.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        # Apply vector field search if vector_query is provided
        if vector_query is not None:
            for field in self.vector_fields:
                if self.vector_matrices[field].shape[1] == vector_query.shape[0]:  # Ensure dimensionality matches
                    vector_sim = cosine_similarity(vector_query.reshape(1, -1), self.vector_matrices[field]).flatten()
                    scores += vector_sim

        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        # Use argpartition to get top num_results indices
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Filter out zero-score results
        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

        return top_docs
