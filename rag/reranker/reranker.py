from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from FlagEmbedding import FlagReranker

from rag.config.config import Config


class Reranker:
    def __init__(self, config: Config):
        self.topk = config["reranker_topk"]
        self.model = FlagReranker(config["rerank_model"])

    def _rerank_single_query(self, query: str, docs: List[str]) -> List[str]:
        # 对每个query的候选文档进行重排
        pairs = [(query, doc) for doc in docs]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs]
        return reranked_docs

    def rerank(self, query_list: List[str], retrieved_list: List[List[str]]) -> List[List[str]]:
        # 多线程重排
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._rerank_single_query, query, docs)
                for query, docs in zip(query_list, retrieved_list)
            ]
            results = [future.result() for future in futures]
        return results
