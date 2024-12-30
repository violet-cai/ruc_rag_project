# -*- coding: utf-8 -*-
import os
import json
import sys

# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from rag.config.config import Config
from rag.database.utils import dense_search, sparse_search
from rag.generator.utils import get_generator
from rag.reranker.utils import get_reranker
from rag.retriever.utils import get_retriever
from rag.evaluator.metrics import (
    F1_Score, Recall_Score, Precision_Score, ExactMatch,Retrieval_Recall,Retrieval_Precision,
    BLEU, Rouge_1, Rouge_2, Rouge_L, LLMJudge, CountToken
)
from rag.evaluator.evaluator import Evaluator
from rag.evaluator.evaluation_data import EvaluationData
from pymilvus import model


import requests 
import certifi
import ssl


path_evaluate_data = 'rag/data/corpus/evaluate_data_beta_filtered.json'

with open(path_evaluate_data, "r", encoding='utf-8') as file:
	evaluate_data = json.loads(file.read())
 
queries = [data.get('query') for data in evaluate_data]
# print(queries)
pred = [data.get('answer') for data in evaluate_data]
# print(pred)
golden_retrieval = []
for relative_doc in evaluate_data:
    single_query_relative_retrieval = relative_doc['reference_docs']
    # single_query_relative_retrieval为list[dict],更改每个dict的key名content为contents
    single_query_relative_retrieval_list = [] 
    for single_doc in single_query_relative_retrieval:
        single_query_relative_retrieval_list.append(single_doc['content'])
    golden_retrieval.append(single_query_relative_retrieval_list)
# print(golden_retrieval)

config = Config()

retriever = get_retriever(config)
retrieved_list = retriever.retrieve(queries)
# retrieved_list_keywords = retriever.retrieve_with_keywords(query) # 关键词检索
# retrieved_list_engine = retriever.search_with_engine(query)
# print(retrieved_list)

# 转为评估格式
eva_retrieval_results = []
for retrieval_result in retrieved_list:
    single_query_retrieval_result = [{"contents":item} for item in retrieval_result]
    eva_retrieval_results.append(single_query_retrieval_result)

config_dict = {
        "dataset_name": "example_dataset",
        "save_dir": "examples/results"
}

def run_evaluation(data):

    # 初始化评估器配置
    config = Config(None,config_dict)

    # 创建评估器实例
    evaluator = Evaluator(config)

    # 调用 Evaluator 执行评估
    results = evaluator.evaluate(data)

    return results

data = EvaluationData(
    # 预测的结果
    pred=pred,
    # 最佳答案
    golden_answers=golden_retrieval,
    # 检索结果
    retrieval_result=eva_retrieval_results
)

# 调用评估接口
scores = run_evaluation(data)

# 打印结果
print(scores)

# reranker = get_reranker(config)
# reranked_list = reranker.rerank(query, retrieved_list)
# print(reranked_list)

# generator = get_generator(config)
# answer = generator.generate(query, reranked_list + retrieved_list_keywords)
# print(answer)