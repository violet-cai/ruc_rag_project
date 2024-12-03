import os
import json
import sys

# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from rag.config.config import Config
from rag.generator.utils import get_generator
from rag.reranker.utils import get_reranker
from rag.retriever.utils import get_retriever
from pymilvus import model


query = "检疫条例有哪些"
# embedding_model = model.hybrid.BGEM3EmbeddingFunction(
#     device="cpu", use_fp16=False, return_sparse=True, return_dense=True, return_colbert_vecs=False
# )

# query_embedding = embedding_model([query])

# res = dense_search(query_embedding=query_embedding["dense"][0], topk=10)
# print(res)

# res = sparse_search(query_embedding=query_embedding["sparse"], topk=10)
# print(res)
config = Config()
retriever = get_retriever(config)
reranker = get_reranker(config)
generator = get_generator(config)

retrieved_list = retriever.retrieve([query])
print(retrieved_list)

reranked_list = reranker.rerank(query, retrieved_list)
print(reranked_list)
