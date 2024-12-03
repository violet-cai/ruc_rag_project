from typing import List, Dict, Any
from .milvus import MilvusClientWrapper
from .milvus import MilvusService
from ..config.config import Config

service = MilvusService(MilvusClientWrapper(host="localhost", port="19530"),"")
config = Config()
    
def dense_search(query_embedding:list, topk=10):
        return service.search(collection_name = config["db_collection_name"],
            query_embedding = query_embedding,
            output_fields = config["db_output_fields"],
            search_by="dense",
            topk = topk)
    

def sparse_search(query_embedding:list, topk=10):
        return service.search(collection_name = config["db_collection_name"],
            query_embedding = query_embedding,
            output_fields = config["db_output_fields"],
            search_by="sparse",
            topk = topk)

def colbert_search(query_embeddings, topk=10):
    pass