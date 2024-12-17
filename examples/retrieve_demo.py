import os
import json
import sys

import torch


# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from rag.config.config import Config
from rag.database.utils import dense_search, sparse_search
from rag.generator.utils import get_generator
from rag.reranker.utils import get_reranker
from rag.retriever.utils import get_retriever
from pymilvus import model


query = "我想了解香港和澳门享受零关税货物原产地标准有哪些？"
embedding_model = model.hybrid.BGEM3EmbeddingFunction(
    devices="cuda:0", return_sparse=True, return_dense=True, return_colbert_vecs=False
)
query = [query]

config = Config()

retriever = get_retriever(config)
retrieved_list = retriever.retrieve(query)
retrieved_list1 = retriever.retrieve_with_keywords(query) # 关键词检索
print(retrieved_list)

reranker = get_reranker(config)
reranked_list = reranker.rerank(query, retrieved_list)
print(reranked_list)

generator = get_generator(config)
answer = generator.generate(query, reranked_list)
print(answer)
