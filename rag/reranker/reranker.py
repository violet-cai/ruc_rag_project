from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from pymilvus import model
from rag.config.config import Config


class Reranker:
    def __init__(self, config: Config):
        self.topk = config["rerank_topk"]
        self.model = model.reranker.BGERerankFunction(config["rerank_model"], device=config["rerank_device"])
        
    
    def rerank(self, query: str, retrieved_list: List[str]) -> List[str]:
        reranked_results = self.model(query,retrieved_list)
        sorted_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)
        reranked_docs = [results.text for results in sorted_results]
        return reranked_docs[: self.topk]
