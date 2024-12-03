import os
import json
import sys

# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.database.utils import dense_search,sparse_search
from pymilvus import model


query = "检疫条例有哪些"
embedding_model = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, return_dense=True, return_colbert_vecs=True)
query_embedding = embedding_model([query])

res = dense_search(query_embedding=query_embedding["dense"][0],topk=10)
print(res)

res = sparse_search(query_embedding=query_embedding["sparse"],topk=10)
print(res)