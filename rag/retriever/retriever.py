from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Literal, Union

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from rag.config.configuration import Config
from rag.database.utils import dense_search, sparse_search, colbert_search


class Retriever:
    def __init__(self, config: Config):
        self.rrf_k = config["rrf_k"]
        self.topk = config["retriever_topk"]
        self.model = BGEM3FlagModel(config["embedding_model"])

    def _get_retrieved_lists(self, query_embedding: Dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"], Union[np.ndarray, List[Union[dict, np.ndarray]]]]) -> \
    Tuple[List[str], List[str], List[str]]:
        dense_retrieved_list = dense_search(query_embedding["dense_vecs"], topk=self.topk * 2)
        sparse_retrieved_list = sparse_search(query_embedding["lexical_weights"], topk=self.topk * 2)
        colbert_retrieved_list = colbert_search(query_embedding["colbert_vecs"], topk=self.topk * 2)
        # 不确定三者检索时间消耗，并行反而可能减速
        # with ThreadPoolExecutor() as executor:
        #     dense_future = executor.submit(dense_search, query_embedding["dense_vecs"], topk=self.topk * 2)
        #     sparse_future = executor.submit(sparse_search, query_embedding["lexical_weights"], topk=self.topk * 2)
        #     colbert_future = executor.submit(colbert_search, query_embedding["colbert_vecs"], topk=self.topk * 2)
        #     dense_retrieved_list = dense_future.result()
        #     sparse_retrieved_list = sparse_future.result()
        #     colbert_retrieved_list = colbert_future.result()
        return dense_retrieved_list, sparse_retrieved_list, colbert_retrieved_list

    def _rrf(self, dense_list: List[str], sparse_list: List[str], colbert_list: List[str]) -> List[str]:
        scores = defaultdict(float)
        for idx, doc in enumerate(dense_list + sparse_list + colbert_list):
            scores[doc] += 1 / (self.rrf_k + idx + 1)
        return [doc for doc, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:self.topk]

    def _retrieve_single_query(self, query: str) -> List[str]:
        query_embedding = self.model.encode_queries(query, return_dense=True, return_sparse=True,
                                                    return_colbert_vecs=True)
        dense_retrieved_list, sparse_retrieved_list, colbert_retrieved_list = self._get_retrieved_lists(query_embedding)
        return self._rrf(dense_retrieved_list, sparse_retrieved_list, colbert_retrieved_list)

    def retrieve(self, query_list: List[str]) -> List[List[str]]:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._retrieve_single_query, query) for query in query_list]
            results = [future.result() for future in futures]
        return results
