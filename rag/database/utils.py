from typing import List, Dict, Any
from .milvus import MilvusClientWrapper
from ..config.config import Config
from pymilvus import model

# embedding_model = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, return_dense=True, return_colbert_vecs=True)
config = Config()
wrapper = MilvusClientWrapper(config)


def dense_search(query_embedding: list, topk=10):
    res_list = wrapper.search(
        collection_name=config["db_collection_name"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_field="dense_vector",
        limit=topk,
        search_params={"metric_type": "L2", "params": {"nprobe": 10}},
    )
    return [res["entity"]["content"] for res in res_list]


def sparse_search(query_embedding: list, topk=10):
    res_list = wrapper.search(
        collection_name=config["db_collection_name"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_field="sparse_vector",
        limit=topk,
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
    )
    return [res["entity"]["content"] for res in res_list]
