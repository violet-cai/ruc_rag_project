from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from FlagEmbedding import FlagReranker
from pymilvus import model
from rag.config.config import Config


class Reranker:
    def __init__(self, config: Config):
        self.topk = config["rerank_topk"]
        self.model = model.reranker.BGERerankFunction(config["rerank_model"], device=config["rerank_device"])
        
    def _rerank_single_query(self, query: str, docs: List[str]) -> List[str]:
        # # 对每个query的候选文档进行重排
        reranked_results = self.model(query,docs)
        sorted_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)
        reranked_docs = [results.text for results in sorted_results]
        return reranked_docs[: self.topk]

    def rerank(self, query_list: List[str], retrieved_list: List[List[str]]) -> List[List[str]]:
        # 多线程重排
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._rerank_single_query, query, docs)
                for query, docs in zip(query_list, retrieved_list)
            ]
            results = [future.result() for future in futures]
        return results
