from collections import defaultdict
from typing import List, Dict, Tuple, Literal, Union

import numpy as np

from rag.config.config import Config
from rag.database.utils import dense_search, sparse_search
from rag.dataprocess.dataloader import load_dataset, get_keywords, get_html, parse_html
from pymilvus import model


class Retriever:
    def __init__(self, config: Config):
        self.rrf_k = config["rrf_k"]  # RRF算法加权参数
        self.topk = config["retriever_topk"]  # topk个文档
        self.fields_to_search = config["fields_to_search"]
        self.model = model.hybrid.BGEM3EmbeddingFunction(
            device="cuda:0", return_sparse=True, return_dense=True, return_colbert_vecs=False
        )  # 采用BGE3模型

    def _get_retrieved_lists(
        self,
        query_embedding: Dict[
            Literal["dense_vecs", "lexical_weights"], Union[np.ndarray, List[Union[dict, np.ndarray]]]
        ],
    ) -> Tuple[List[dict], List[dict]]:
        # 分别检索三种检索方法
        dense_retrieved_list = dense_search(query_embedding=query_embedding["dense"][0], topk=self.topk * 3)  # 密集向量
        sparse_retrieved_list = sparse_search(query_embedding=query_embedding["sparse"], topk=self.topk)  # 稀疏向量
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

    def _deduplicate_merge(self, dense_retrieved_list, sparse_retrieved_list):
        dense_set = set(dense_retrieved_list)
        sparse_set = set(sparse_retrieved_list)
        merged_set = dense_set.union(sparse_set)
        merged_list = list(merged_set)
        return merged_list

    def retrieve(self, query_list: List[str]) -> List[List[str]]:
        retrieved_docs = []
        for query in query_list:
            query_embedding = self.model([query])
            dense_retrieved_list, sparse_retrieved_list = self._get_retrieved_lists(query_embedding)
            docs = self._deduplicate_merge(dense_retrieved_list, sparse_retrieved_list)
            retrieved_docs.append(docs)
        return retrieved_docs

    def retrieve_with_keywords(self, query_list: List[str]) -> List[List[str]]:
        retrieved_docs1 = []
        retrieved_docs2 = []
        dataset1 = load_dataset("dataset")
        dataset2 = load_dataset("qa")
        fields_to_search = self.fields_to_search
        for query in query_list:
            keywords_with_weights = get_keywords(query)
            # 权重前5的关键词用于检索
            top_weighted_keywords = [kw for kw, weight in keywords_with_weights[:5]]

            keyword_docs1 = []
            for document in dataset1:
                matched_keywords_count = 0
                for keyword in top_weighted_keywords:
                    if any(keyword in str(document.get(field, "")) for field in fields_to_search):
                        matched_keywords_count += 1
                # 如果关键词匹配度超过50%,则将文档添加到检索结果
                if matched_keywords_count / len(top_weighted_keywords) >= 0.5:
                    doc_content = document.get("content", "")
                    keyword_docs1.append((doc_content, matched_keywords_count))
            keyword_docs1.sort(key=lambda x: x[1], reverse=True)
            retrieved_docs1.append([doc[0] for doc in keyword_docs1])

            keyword_docs2 = []
            for document in dataset2:
                matched_keywords_count = 0
                for keyword in top_weighted_keywords:
                    if any(keyword in str(document.get(field, "")) for field in fields_to_search):
                        matched_keywords_count += 1
                # 如果关键词匹配度超过50%,则将文档添加到检索结果
                if matched_keywords_count / len(top_weighted_keywords) >= 0.5:
                    doc_content = document.get("content", "")
                    keyword_docs2.append((doc_content, matched_keywords_count))
            keyword_docs2.sort(key=lambda x: x[1], reverse=True)
            retrieved_docs2.append([doc[0] for doc in keyword_docs2])

        return retrieved_docs1[: self.topk] + retrieved_docs2[: self.topk]

    def retrieve_with_engine(self, query: str) -> List[str]:
        search_content = get_html(query=query, mode="search")
        search_results = parse_html(search_content, "search")

        retrieved_docs = []
        for result in search_results:
            url = result["link"]
            title = result["title"]
            doc_content = get_html(url=url, mode="extract")
            doc = parse_html(doc_content, "extract")
            if doc:
                retrieved_docs.append({"title": title, "doc": doc})
            else:
                continue
        return retrieved_docs
