from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Literal, Union

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from rag.config.config import Config
from rag.database.utils import dense_search, sparse_search
from rag.dataprocess.dataloader import get_datasets, get_keywords, bing_search, baidu_search, get_docs
from pymilvus import model



class Retriever:
    def __init__(self, config: Config):
        self.rrf_k = config["rrf_k"]    # RRF算法加权参数
        self.topk = config["retriever_topk"]    # topk个文档
        self.fields_to_search = config['fields_to_search']
        self.collection_name = config["db_collection_name"]
        self.model = model.hybrid.BGEM3EmbeddingFunction(
            device="cuda:0", return_sparse=True, return_dense=True, return_colbert_vecs=False
        )   # 采用BGE3模型

    def _get_retrieved_lists(
        self,
        query_embedding: Dict[
            Literal["dense_vecs", "lexical_weights"], Union[np.ndarray, List[Union[dict, np.ndarray]]]
        ],
    ) -> Tuple[List[dict], List[dict]]:
        # 分别检索三种检索方法
        dense_retrieved_list = dense_search(query_embedding=query_embedding["dense"][0], topk=self.topk * 3, collection_name=self.collection_name)    # 密集向量
        sparse_retrieved_list = sparse_search(query_embedding=query_embedding["sparse"], topk=self.topk, collection_name=self.collection_name)    # 稀疏向量
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


    def retrieve_with_keywords(self, query_list: List[str]) -> List[List[str]]:
        retrieved_docs = []
        datasets = get_datasets()
        
        for query in query_list:
            keywords_with_weights = get_keywords(query)
            # 权重前5的关键词用于检索
            top_weighted_keywords = [kw for kw, weight in keywords_with_weights[:5]]
            for dataset in datasets:
                retrieved_docs.append(self.keyword_retrieval(dataset,top_weighted_keywords))
                
        return retrieved_docs
    
    def keyword_retrieval(self, dataset:dict,top_weighted_keywords:list) -> List[List[str]]:
        fields_to_search = self.fields_to_search
        keyword_docs = []
        for document in dataset:
            matched_keywords_count = 0
            for keyword in top_weighted_keywords:
                if any(keyword in str(document.get(field, "")) for field in fields_to_search):
                    matched_keywords_count += 1
            # 如果关键词匹配度超过50%,则将文档添加到检索结果
            if matched_keywords_count / len(top_weighted_keywords) >= 0.2:
                doc_content = document.get("content", "") 
                ans_content = document.get("answer", "")
                if ans_content:
                    doc_content = doc_content + document.get("answer", "")
                keyword_docs.append((doc_content, matched_keywords_count))
        keyword_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in keyword_docs[:self.topk]]
    
    def bing_retrieval(self, query_list: List[str]) -> List[List[str]]:
        retrieved_docs = []
        for query in query_list:
            doc = self.retrieve_with_engine(query,'bing')
            retrieved_docs.append(doc)
        return retrieved_docs
    
    def baidu_retrieval(self, query_list: List[str]) -> List[List[str]]:
        retrieved_docs = []
        for query in query_list:
            doc = self.retrieve_with_engine(query,'baidu')
            retrieved_docs.append(doc)
        return retrieved_docs
    
    def retrieve_with_engine(self, query: str, mode='bing') -> List[str]:
        if mode == 'bing':
            search_results = bing_search(query)
        elif mode == 'baidu':
            search_results = baidu_search(query)
        if search_results:
            retrieved_docs = get_docs(search_results)
        
        return retrieved_docs[:self.topk]
    
