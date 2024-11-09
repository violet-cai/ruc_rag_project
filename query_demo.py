import json

from rag.query.query import parse_query

query = "中国特色社会主义有什么优点，我们为什么要坚持走特色社会主义道路，特色社会主义道路带给我们什么"
query_list = []
queries_json = parse_query(query)
queries = json.loads("[" + queries_json + "]")
for item in queries:
    query_key = next((key for key in item.keys() if key.startswith("query")), None)
    if query_key:
        query = item[query_key]
        references_list = [item[key] for key in item.keys() if key.startswith("reference")]
        query_list.append({"query": query, "references": references_list})
for query in query_list:
    print(query)

# TODO 考虑如何使用query和references_list，可能要进行对比实验
"""
1.只使用references检索的数据 
2.使用references和query混合搜索，把结果都直接装进去 
3.计算retrieval得到的score再综合排序筛选
"""
