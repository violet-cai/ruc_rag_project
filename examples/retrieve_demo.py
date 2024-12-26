import os
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


query = "对原产于美国的物品有哪些反倾销措施"
embedding_model = model.hybrid.BGEM3EmbeddingFunction(
    devices="cuda:0", return_sparse=True, return_dense=True, return_colbert_vecs=False
)
query = [query]

config = Config()

retriever = get_retriever(config)
retrieved_list = retriever.retrieve(query)
retrieved_list_keywords = retriever.retrieve_with_keywords(query)
retrieved_list_bing = retriever.bing_retrieval(query)
retrieved_list_baidu = retriever.baidu_retrieval(query)
# print(len(retrieved_list[0]))
# print(retrieved_list_keywords[0])
# print(retrieved_list_engine[0])

reranker = get_reranker(config)
reranked_list = reranker.rerank(query, retrieved_list)

generator = get_generator(config)
answer = generator.generate(query, reranked_list)
print(answer)
