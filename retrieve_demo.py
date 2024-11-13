import concurrent.futures
import json

from rag.config.config import Config
from rag.generator import generator
from rag.query.query import parse_query
from rag.retriever.retriever import BM25Retriever, DenseRetriever, Reranker


def retrieve(query_list, text_list, bm25_tokenized, bm25_model, bge_model, vector_db):
    retriever1 = BM25Retriever(config, bm25_tokenized, bm25_model)
    # TODO query_instruction看看怎么使用，以及检索混合分数而不是直接融合
    retriever2 = DenseRetriever(config, None, bge_model, vector_db)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(retriever1.retrieve, query_list, text_list)
        future2 = executor.submit(retriever2.retrieve, query_list, text_list)
        retriever1_result = future1.result()
        retriever2_result = future2.result()
    merged_results = []
    for a, b in zip(retriever1_result, retriever2_result):
        unique_texts = set(a).union(set(b))
        merged_texts = list(unique_texts)
        merged_results.append(merged_texts)
    return merged_results


# TODO后续可能赋予改写和原始问题权重，这可能要连带改写retriever
def query_rephrase(query):
    query_list = []
    queries_json = parse_query(query)
    queries = json.loads("[" + queries_json + "]")
    for item in queries:
        query_key = next((key for key in item.keys() if key.startswith("query")), None)
        if query_key:
            query = item[query_key]
            references_list = [item[key] for key in item.keys() if key.startswith("reference")]
            query_list.append({"query": query, "references": references_list})
    new_query_list = []
    for item in query_list:
        new_query = [item["query"]]
        for reference in item["references"]:
            new_query.append(reference)
        new_query_list.append(new_query)
    return new_query_list


config = Config()
# 获取数据
info_dict = {
    "document type": "",  # 法规类型
    "category": "",  # 法规类别
    "announcement_number": "",  # 文号
    "issuing authority": "",  # 发布机构
    "issue date": "",  # 发布日期
    "effective_date": "",  # 生效日期
    "status": "",  # 效力
    "remarks": "",  # 效力说明
    "title": "",  # 标题
    "content": "",  # 内容
    "appendix": ""  # 附件
}
json_path = "data/corpus/regulations.json"
with open(json_path, "r") as f:
    json_data = json.load(f)
texts = []
for data in json_data:
    texts.append(data["content"])

# 修改query
my_query = "我国现在是否允许塞拉利昂野生水产品进口"
my_query_list = query_rephrase(my_query)

# 检索
bm25_tokenized_file = 'data/database/tokenized_text.pkl'
bm25_model_path = 'data/database/bm25_model.pkl'
bge_model_path = 'model/BAAI/bge-base-zh-v1.5'
vector_db_file = 'data/database/vector_db.index'
retrieved_result = retrieve(my_query_list, texts, bm25_tokenized_file, bm25_model_path, bge_model_path, vector_db_file)

# 重排
model_path = 'model/BAAI/bge-reranker-base'
reranker = Reranker(config, model_path)
rerank_result = reranker.rerank(my_query_list, retrieved_result)
print(my_query_list, rerank_result)

# 生成
generator.generate(my_query_list, rerank_result)
