from typing import List, Dict, Any
from .milvus import MilvusClientWrapper
from .milvus import MilvusService


def dense_search(service:MilvusService, collection_name, query_embedding:list, output_fields, topk=10):
    return service.search(collection_name = collection_name,
            query_embedding = query_embedding,
            output_fields = output_fields,
            search_by="dense",
            topk = topk)
    

def sparse_search(service:MilvusService, collection_name, query_embedding:list, output_fields, topk=1):
        return service.search(collection_name = collection_name,
            query_embedding = query_embedding,
            output_fields = output_fields,
            search_by="sparse",
            topk = topk)


def colbert_search(query_embeddings, topk=10):
    pass