from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Literal, Union

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from rag.config.config import Config
from rag.database.utils import dense_search, sparse_search
from pymilvus import model


class Retriever:
    def __init__(self, config: Config):
        self.rrf_k = config["rrf_k"]
        self.topk = config["retriever_topk"]
        self.model = model.hybrid.BGEM3EmbeddingFunction(
            device="cuda:0", return_sparse=True, return_dense=True, return_colbert_vecs=False
        )

    def _get_retrieved_lists(
        self,
        query_embedding: Dict[
            Literal["dense_vecs", "lexical_weights"], Union[np.ndarray, List[Union[dict, np.ndarray]]]
        ],
    ) -> Tuple[List[dict], List[dict]]:
        # 分别检索三种检索方法
        dense_retrieved_list = dense_search(query_embedding=query_embedding["dense"][0], topk=self.topk * 3)
        sparse_retrieved_list = sparse_search(query_embedding=query_embedding["sparse"], topk=self.topk)
        # 不确定检索时间消耗，并行反而可能减速
        # with ThreadPoolExecutor() as executor:
        #     dense_future = executor.submit(dense_search, query_embedding["dense_vecs"][0], topk=self.topk * 5)
        #     sparse_future = executor.submit(sparse_search, query_embedding["lexical_weights"], topk=self.topk * 2)
        #     dense_retrieved_list = dense_future.result()
        #     sparse_retrieved_list = sparse_future.result()
        return dense_retrieved_list, sparse_retrieved_list

    def _rrf(self, dense_list: List[dict], sparse_list: List[dict]) -> List[dict]:
        # 混合检索RRF分数
        scores = defaultdict(float)
        for idx, doc in enumerate(dense_list + sparse_list):
            scores[doc] += 1 / (self.rrf_k + idx + 1)
        return [doc for doc, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)][: self.topk]

    def retrieve(self, query_list: List[str]) -> List[List[str]]:
        retrieved_docs = []
        for query in query_list:
            query_embedding = self.model([query])
            dense_retrieved_list, sparse_retrieved_list = self._get_retrieved_lists(query_embedding)
            docs = self._rrf(dense_retrieved_list, sparse_retrieved_list)
            retrieved_docs.append(docs)
        return retrieved_docs
