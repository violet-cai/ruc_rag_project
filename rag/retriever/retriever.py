import os
import pickle

import faiss
import jieba
import numpy as np
from FlagEmbedding import FlagModel, FlagReranker
from rank_bm25 import BM25Okapi


def split_origin_rewritten_query(top_k, queries):
    original_query = [queries[0]]
    rewritten_queries = queries[1:]
    original_len = top_k // len(queries) + top_k % len(queries)
    rewritten_len = top_k // len(queries)
    original_len = int(original_len)
    rewritten_len = int(rewritten_len)
    return original_query, rewritten_queries, original_len, rewritten_len


class BM25Retriever:
    def __init__(self, config, tokenized_text_file, bm25_model_file):
        self.tokenized_text = None
        self.model = None
        self.top_k = 10
        self.tokenized_text_file = tokenized_text_file
        self.bm25_model_file = bm25_model_file

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

    def retrieve(self, query_list, corpus_split):
        if self.model is None or self.tokenized_text is None:
            if not self._load_tokenized_text():
                self.tokenized_text = [list(jieba.cut(doc)) for doc in corpus_split]
                self._save_tokenized_text()
            if not self._load_model():
                # TODO 这里不一定适用BM25Okapi，后续evaluate时可以对比BM25+、BM25L
                self.model = BM25Okapi(self.tokenized_text)
                self._save_model()
        results = []
        for queries in query_list:
            original_query, rewritten_queries, original_len, rewritten_len = split_origin_rewritten_query(
                self.top_k, queries)
            tokenized_original_query = list(jieba.cut(original_query[0]))
            scores = self.model.get_scores(tokenized_original_query)
            top_k_indices = np.argsort(-scores)[:original_len]
            original_top_texts = [(corpus_split[i]) for i in top_k_indices]
            rewritten_top_texts = []
            for query in rewritten_queries:
                tokenized_query = list(jieba.cut(query))
                scores = self.model.get_scores(tokenized_query)
                top_k_indices = np.argsort(-scores)[:rewritten_len]
                top_texts = [(corpus_split[i]) for i in top_k_indices]
                rewritten_top_texts.append(top_texts)
            all_texts = set(original_top_texts)
            for texts in rewritten_top_texts:
                all_texts.update(set(texts))
            final_texts = list(all_texts)
            results.append(final_texts)
        return results


class DenseRetriever:
    def __init__(self, config, query_instruction, model_path, vector_db_file):
        self.model = FlagModel(model_path, query_instruction_format=query_instruction, use_fp16=True)
        self.top_k = 10
        self.db_file = vector_db_file
        self.vectordb = None

    def _load_vector_db(self):
        if os.path.exists(self.db_file):
            self.vectordb = faiss.read_index(self.db_file)
            return True
        else:
            return False

    def _save_vector_db(self, corpus_split):
        text_embeddings = self.model.encode(corpus_split)
        dimension = text_embeddings.shape[1]
        self.vectordb = faiss.IndexFlatIP(dimension)
        self.vectordb.add(text_embeddings.astype('float32'))
        faiss.write_index(self.vectordb, self.db_file)

    def retrieve(self, query_list, corpus_split):
        if self.vectordb is None:
            if not self._load_vector_db():
                self._save_vector_db(corpus_split)
                self._load_vector_db()
        results = []
        for queries in query_list:
            original_query, rewritten_queries, original_len, rewritten_len = split_origin_rewritten_query(
                self.top_k, queries)
            original_query_embedding = self.model.encode_queries(original_query)
            _, original_top_k_indices = self.vectordb.search(original_query_embedding.astype('float32'), k=original_len)
            original_top_texts = [(corpus_split[j]) for j in original_top_k_indices[0]]
            rewritten_query_embeddings = self.model.encode_queries(rewritten_queries)
            _, rewritten_top_k_indices = self.vectordb.search(rewritten_query_embeddings.astype('float32'),
                                                              k=rewritten_len)
            rewritten_top_texts = [[(corpus_split[j]) for j in indices] for indices in rewritten_top_k_indices]
            all_texts = set(original_top_texts)
            for texts in rewritten_top_texts:
                all_texts.update(set(texts))
            final_texts = list(all_texts)
            results.append(final_texts)
        return results


class Reranker:
    def __init__(self, config, model_path):
        self.model = FlagReranker(model_path, use_fp16=True)
        self.top_k = 2

    # TODO rerank模型如何处理original_query和rewritten_queries会更好
    def rerank(self, query_list, retrieved_list):
        results = []
        for queries, texts in zip(query_list, retrieved_list):
            original_query = queries[0]
            rewritten_queries = queries[1:]
            combined_queries = [original_query] + rewritten_queries
            inputs = []
            for query in combined_queries:
                inputs.extend([[query, text] for text in texts])
            scores = self.model.compute_score(inputs)
            scores = np.array(scores).reshape(len(combined_queries), len(texts))
            average_scores = np.mean(scores, axis=0)
            sorted_indices = (-average_scores).argsort()[:self.top_k]
            top_results = [texts[i] for i in sorted_indices]
            results.append(top_results)
        return results
