from pymilvus import MilvusClient, Collection, CollectionSchema, FieldSchema, DataType
from .logger import db_logger
from typing import Union

#  客户端
class MilvusClientWrapper:
    def __init__(self, host='localhost', port='19530'):
        self.client = MilvusClient(host=host, port=port)
        db_logger.info(f"Milvus 连接到 {host}:{port}")

    def create_collection(self, collection_name, schema):
        try:
            if self.client.has_collection(collection_name):
                db_logger.info(f"collection: {collection_name} 已存在")
                return
            self.client.create_collection(collection_name=collection_name, schema=schema)
            db_logger.info(f"collection: {collection_name} 创建成功")
        except Exception as e:
            db_logger.error(f"创建 collection 失败: {e}")
            raise

    def create_index(self, collection_name:str,index_field_names:list):
        try:
            index_params = self.client.prepare_index_params()
            
            for field_name in index_field_names:
                # 稀疏向量索引
                if field_name == "sparse_vector":
                    index_params.add_index(
                        field_name=field_name,
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"nlist": 1024}
                    )
                # 稠密向量索引
                else:
                    index_params.add_index(
                        field_name=field_name,
                        index_type="IVF_FLAT",
                        metric_type="L2",
                        params={"nlist": 1024}
                    )
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            db_logger.info(f"collection: {collection_name} 的索引创建成功")
        except Exception as e:
            db_logger.error(f"创建索引失败: {e}")
            raise

    def insert_data(self, collection_name, data):
        try:
            res = self.client.insert(collection_name=collection_name, data=data)
            db_logger.info(f"表 {collection_name} 插入数据: {res['insert_count']} 条")
        except Exception as e:
            db_logger.error(f"插入数据失败: {e}")
            raise

    def search(self, collection_name:str, query_embedding:list, search_params:dict, limit:int, output_fields:list, search_field:str):
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                output_fields=output_fields,
                anns_field=search_field,
                search_params=search_params,
                limit=limit
            )
            return results[0]
        except Exception as e:
            db_logger.error(f"搜索失败: {e}")
        
# 服务端        
class MilvusService:
    def __init__(self, client: MilvusClientWrapper, embedding_model):
        self.client = client
        self.embedding_model = embedding_model
        self.vector_dim = 1024

    def _create_milvus_schema(self, info_dict:dict):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # FieldSchema(name="colbert_vector", dtype=DataType.FLOAT_VECTOR,dim=self.vector_dim)
        ]
        for key, value in info_dict.items():
            if key == "content":
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=65535, is_primary=False))
            else:
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=255, is_primary=False))
        schema = CollectionSchema(fields=fields, description="Custom collection schema")
        return schema

    def create_collection(self, collection_name:str, info_dict:dict, index_field_names:list):
        schema = self._create_milvus_schema(info_dict)
        self.client.create_collection(collection_name, schema)
        self.client.create_index(collection_name, index_field_names)

    def insert_or_update(self, collection_name:str, chunk_list: list[dict]):
        model = self.embedding_model
        embeddings = [model.encode_queries([item["content"]]) for item in chunk_list]
        data = chunk_list
        for i, item in enumerate(data):
            item["dense_vector"] = embeddings[i]["dense"][0]
            item["sparse_vector"] = embeddings[i]["sparse"]
            # item["colbert_vector"] = embeddings[i]["colbert_vecs"]
            if "id" in item:
                del item["id"]
        self.client.insert_data(collection_name, data)
 
    def search(self, collection_name:str, query_embedding:list, topk:int, search_by:str, output_fields:list):
        if search_by == "dense":
            search_params = {"metric_type": "L2","params": {"nprobe": 10}}
            search_field = "dense_vector"
        elif search_by == "sparse":
            search_params = {"metric_type": "IP","params": {"drop_ratio_search": 0.2}}
            search_field = "sparse_vector"
        # elif: colbert向量
        return self.client.search(collection_name, query_embedding, search_params, topk, output_fields,search_field)
    
    
