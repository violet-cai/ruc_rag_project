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
from rag.dataprocess.queryupdater import get_new_query_1,get_new_query_2
from pymilvus import model


query = "如果我们在准备材料的过程中遇到问题，可以联系谁呢？"
embedding_model = model.hybrid.BGEM3EmbeddingFunction(
    devices="cuda:0", return_sparse=True, return_dense=True, return_colbert_vecs=False
)
query = [query]

historys = [
        {
            "query":"您好，我想了解一下关于中国澳门制造的保健食品在华注册备案的事宜。",
            "answer":"您好！根据相关规定，对于进口保健食品等18类食品的境外生产企业，采用“官方推荐注册”模式。也就是说，澳门特别行政区的主管当局需要对推荐注册的企业进行审核检查，并向海关总署推荐注册。"
        },
        {
            "query":"那是不是只有生产企业才能提交注册备案呢？如果是一般的贸易企业或者代理商可以吗？",
            "answer":"根据规定，必须是由所在国家或地区的主管当局推荐并提交申请材料，因此一般贸易企业或代理商本身没有直接提交注册备案的权利。不过，这些企业可以协助生产企业准备相关材料。"
        },
        {
            "query":"明白了，那具体需要提交哪些材料呢？",
            "answer":"主要包括：所在国家或地区主管当局的推荐函、企业名单与企业注册申请书、企业的身份证明文件（如营业执照）、主管当局的声明以及对相关企业的审查报告等。"
        },
    ]
history = {}
config = Config()

query = [get_new_query_1(historys, query[-1])]
# query = [get_new_query_2(historys, query[-1])]
retriever = get_retriever(config)
retrieved_list = retriever.retrieve(query)

history["query"] = query

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

history["answer"] = answer
historys.append(history)
print(answer)
