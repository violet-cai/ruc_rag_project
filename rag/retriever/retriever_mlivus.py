import os
import pickle

from pymilvus import MilvusClient,CollectionSchema
from pymilvus import model
from rag.milvus.milvus_client import CustomMilvusClient
from rag.milvus.logger import db_logger

class MilvusRetriever:
    def __init__(self, host='localhost', port='19530', client = None, embedding_model = None): 
        self.client = client if client is not None else CustomMilvusClient(host=host, port=port).client
        self.embedding_model = embedding_model if embedding_model is not None else self._enable_embedding()

    def denseQuery(self, collection_name, query_list, output_fields,top_k = 5):
        
        if(isinstance(query_list,str)):
            query_list = [query_list]
            
        query_embeddings = self.embedding_model.encode_queries(query_list)
        
        try:
            res = self.client.search(collection_name = collection_name,
                                    data = query_embeddings['dense'],
                                    output_fields = output_fields,
                                    search_params = {"metric_type": "L2"},
                                    limit = top_k)
        except Exception as e:
            db_logger.error(f"检索失败, 错误信息: {e}")
        return self._res_to_string(res,output_fields=output_fields)

    
    # 选择embedding 模型
    # 此处默认使用BGEM3EmbeddingFunction，后续按需求增加新模型
    def _enable_embedding(self):
        embedding_fn = model.hybrid.BGEM3EmbeddingFunction()
        db_logger.info(f"embedding模型: {embedding_fn}")
        return embedding_fn
    
    # 将res的title content字段连接为字符串：
    def _res_to_string(self, res,output_fields) -> str:
        knowledge = ""
        for i,res_i in enumerate(res):
            knowledge += "====================\n"
            knowledge += f"The response of {i+1} query:\n"
            for j,row in enumerate(res_i):
                knowledge += f"\tTop {j+1}:\n"
                for field in output_fields:
                    knowledge += f"\t{field}: {row['entity'][field]}\n"
        return knowledge
        


    

