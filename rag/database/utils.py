from typing import List, Dict, Any
from .milvus import MilvusClientWrapper
from .milvus import MilvusService
from ..config.config import Config
from pymilvus import model

embedding_model = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, return_dense=True, return_colbert_vecs=True)
service = MilvusService(
    client=MilvusClientWrapper(),
    embedding_model=embedding_model,
)
config = Config()


def dense_search(query_embedding: list, topk=10):
    return service.search(
        collection_name=config["db_collection_name"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_by="dense",
        topk=topk,
    )


def sparse_search(query_embedding: list, topk=10):
    return service.search(
        collection_name=config["db_collection_name"],
        query_embedding=query_embedding,
        output_fields=config["db_output_fields"],
        search_by="sparse",
        topk=topk,
    )
