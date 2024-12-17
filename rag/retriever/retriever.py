from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Literal, Union

import numpy as np
import re
from FlagEmbedding import BGEM3FlagModel

from rag.config.config import Config
from rag.database.utils import dense_search, sparse_search
from rag.data.dataloader import load_dataset, load_stopwords, get_dict_path
from pymilvus import model
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba  

class Retriever:
    def __init__(self, config: Config):
        self.rrf_k = config["rrf_k"]    # RRF算法加权参数
        self.topk = config["retriever_topk"]    # topk个文档
        self.fields_to_search = config['fields_to_search']
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
        dense_retrieved_list = dense_search(query_embedding=query_embedding["dense"][0], topk=self.topk * 3)    # 密集向量
        sparse_retrieved_list = sparse_search(query_embedding=query_embedding["sparse"], topk=self.topk)    # 稀疏向量
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
        dataset = load_dataset('dataset')
        fields_to_search = self.fields_to_search
        for query in query_list:
            keywords_with_weights = self._get_keywords(query)
            # 权重前5的关键词用于检索
            top_weighted_keywords = [kw for kw, weight in keywords_with_weights[:5]]
            
            keyword_docs = []
            for document in dataset:
                matched = False
                for keyword in top_weighted_keywords:
                    if any(keyword in str(document.get(field, "")) for field in fields_to_search):
                        matched = True
                        break
                if matched:
                    doc_content = document.get("content","")
                    keyword_docs.append(doc_content)

            retrieved_docs.append(keyword_docs)
            
        return retrieved_docs
    
    
    def _get_keywords(self, query: str) -> List[str]:
        
        dict_path = get_dict_path()
        jieba.load_userdict(dict_path)
        new_data = ''.join(re.findall('[\u4e00-\u9fa5]+', query))
        stopwords = load_stopwords()
        
        seg_list = jieba.lcut(new_data)
        # 去除停用词并且去除单字
        filtered_words = [word for word in seg_list if word not in stopwords and len(word) > 1]
        # 使用TF-IDF对关键词进行打分
        tfidf_vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(filtered_words)])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        keywords_with_weights = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        keywords_with_weights.sort(key=lambda x: x[1], reverse=True)  
        
        return keywords_with_weights
    
    

        