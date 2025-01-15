from typing import List, Dict, Any
from .milvus import MilvusClientWrapper
from ..config.config import Config
from pymilvus import model

# embedding_model = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, return_dense=True, return_colbert_vecs=True)
config = Config()
wrapper = MilvusClientWrapper(config)


def dense_search(query_embedding: list, topk=10):
    res_list_regu = wrapper.search(
        collection_name=config["db_collection_name_regu"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_field="dense_vector",
        limit=topk,
        search_params=config["db_dense_search_params"],
    )
    res_list_qa = wrapper.search(
        collection_name=config["db_collection_name_qa"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_field="dense_vector",
        limit=topk,
        search_params=config["db_dense_search_params"],
    )
    res_regu = [res["entity"]["content"] for res in res_list_regu]
    res_qa = [res["entity"]["content"] for res in res_list_qa]
    res_list = res_regu + res_qa
    return res_list


def sparse_search(query_embedding: list, topk=10):
    res_list_regu = wrapper.search(
        collection_name=config["db_collection_name_regu"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_field="sparse_vector",
        limit=topk,
        search_params=config["db_sparse_search_params"],
    )
    res_list_qa = wrapper.search(
        collection_name=config["db_collection_name_qa"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_field="sparse_vector",
        limit=topk,
        search_params=config["db_sparse_search_params"],
    )
    res_regu = [res["entity"]["content"] for res in res_list_regu]
    res_qa = [res["entity"]["content"] for res in res_list_qa]
    res_list = res_regu + res_qa
    return res_list
