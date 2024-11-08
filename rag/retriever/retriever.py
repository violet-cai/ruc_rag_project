import os
import pickle

import faiss
import jieba
import numpy as np
from FlagEmbedding import FlagModel, FlagReranker
from rank_bm25 import BM25Okapi


# retriever_model_path = 'model/BAAI/bge-base-zh-v1.5'
# reranker_model_path = 'model/BAAI/bge-reranker-v2-m3'


class BM25Retriever:
    def __init__(self, config):
        self.tokenized_text = None
        self.model = None
        self.top_k = 8
        self.tokenized_text_file = 'data/database/tokenized_text.pkl'
        self.bm25_model_file = 'data/database/model.pkl'

    def _save_tokenized_text(self):
        with open(self.tokenized_text_file, 'wb') as f:
            pickle.dump(self.tokenized_text, f)

    def _load_tokenized_text(self):
        if os.path.exists(self.tokenized_text_file):
            with open(self.tokenized_text_file, 'rb') as f:
                self.tokenized_text = pickle.load(f)
            return True
        else:
            return False

    def _save_model(self):
        with open(self.bm25_model_file, 'wb') as f:
            pickle.dump(self.model, f)

    def _load_model(self):
        if os.path.exists(self.bm25_model_file):
            with open(self.bm25_model_file, 'rb') as f:
                self.model = pickle.load(f)
            return True
        else:
            return False

    def query(self, query_list, corpus_split):
        if self.model is None or self.tokenized_text is None:
            if not self._load_tokenized_text():
                self.tokenized_text = [list(jieba.cut(doc)) for doc in corpus_split]
                self._save_tokenized_text()
            if not self._load_model():
                self.model = BM25Okapi(self.tokenized_text)
                self._save_model()
        results = []
        for query in query_list:
            tokenized_query = list(jieba.cut(query))
            scores = self.model.get_scores(tokenized_query)
            top_k_indices = np.argsort(-scores)[:self.top_k]
            top_k_texts = [(corpus_split[i]) for i in top_k_indices]
            results.append(top_k_texts)
        return results


class DenseRetriever:
    def __init__(self, config, query_instruction, model_path):
        self.model = FlagModel(model_path, query_instruction_format=query_instruction, use_fp16=True)
        self.top_k = 8
        self.db_file = 'data/database/vector_db.index'
        self.vectordb = None

    def _load_vector_db(self):
        if os.path.exists(self.db_file):
            self.vectordb = faiss.read_index(self.db_file)
            return True
        else:
            return False

    def query(self, query_list, corpus_split):
        query_embeddings = self.model.encode_queries(query_list)
        if self.vectordb is None:
            if not self._load_vector_db():
                text_embeddings = self.model.encode(corpus_split)
                dimension = text_embeddings.shape[1]
                self.vectordb = faiss.IndexFlatIP(dimension)
                self.vectordb.add(text_embeddings.astype('float32'))
                faiss.write_index(self.vectordb, self.db_file)
        scores, top_k_indices = self.vectordb.search(query_embeddings.astype('float32'), k=self.top_k)
        results = []
        for indices in top_k_indices:
            top_texts = [(corpus_split[j]) for j in indices]
            results.append(top_texts)
        return results


class Reranker:
    def __init__(self, config, model_path):
        self.model = FlagReranker(model_path, use_fp16=True)
        self.top_k = 2

    def rerank(self, query_list, retrieved_list):
        results = []
        for query, texts in zip(query_list, retrieved_list):
            inputs = [[query, text] for text in texts]
            scores = self.model.compute_score(inputs)
            scores = np.array(scores)
            sorted_indices = (-scores).argsort()[:self.top_k]
            sorted_indices = sorted_indices.flatten().tolist()
            top_results = [texts[i] for i in sorted_indices]
            results.append(top_results)
        return results
