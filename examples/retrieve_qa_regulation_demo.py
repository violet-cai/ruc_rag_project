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
config['db_collection_name'] = 'qa'
FQA_retriever = get_retriever(config)
FQA_retrieved_list = FQA_retriever.retrieve(query)

config['db_collection_name'] = 'regulation'
refer_retriever = get_retriever(config)
refer_retrieved_list = refer_retriever.retrieve(query)

reranker = get_reranker(config)
FQA_reranked_list = reranker.rerank(query, FQA_retrieved_list)
refer_reranked_list = reranker.rerank(query, refer_retrieved_list)

generator = get_generator(config)
answer = generator.generate(query, FQA_reranked_list=FQA_reranked_list, refer_reranked_list=refer_reranked_list)
print(answer)