from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from .logger import db_logger

from rag.config.config import Config


#  客户端
class MilvusClientWrapper:
    def __init__(self, config: Config):
        self.client = MilvusClient(config["db_uri"])
        self.vector_dim = config["db_embedding_dim"]
        self.config = config
        db_logger.info(f"Milvus 连接")

    def create_collection(self, collection_name: str, info_dict: dict):
        try:
            if self.client.has_collection(collection_name):
                db_logger.info(f"collection: {collection_name} 已存在")
                return
            schema = self._create_milvus_schema(info_dict)
            self.client.create_collection(collection_name=collection_name, schema=schema)
            db_logger.info(f"collection: {collection_name} 创建成功")
        except Exception as e:
            db_logger.error(f"创建 collection 失败: {e}")
            raise

    def _create_milvus_schema(self, info_dict: dict):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # FieldSchema(name="colbert_vector", dtype=DataType.FLOAT_VECTOR,dim=self.vector_dim)
        ]
        for key, value in info_dict.items():
            if key == self.config["db_content_key"]:
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=5096, is_primary=False))
            elif key == self.config["db_text_id"]:
                fields.append(FieldSchema(name=key, dtype=DataType.INT64, is_primary=False, auto_id=False))
            elif key == self.config["db_metadata_key"]:
                fields.append(FieldSchema(name=key, dtype=DataType.JSON, is_primary=False))
            else:
                if isinstance(value, str):
                    fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=2048, is_primary=False))
                else:
                    fields.append(FieldSchema(name=key, dtype=DataType.JSON, is_primary=False))
        schema = CollectionSchema(fields=fields, description="Custom collection schema")
        return schema

    def set_index(self, collection_name: str):
        try:
            index_params = self.client.prepare_index_params()
            for field_name in self.config["db_index_fields"]:
                # 稀疏向量索引
                if field_name == "sparse_vector":
                    index_params.add_index(
                        field_name=field_name,
                        index_type=self.config["db_sparse_index_type"],
                        metric_type=self.config["db_sparse_metric_type"],
                        params=self.config["db_sparse_params"],
                    )
                # 稠密向量索引
                else:
                    index_params.add_index(
                        field_name=field_name,
                        index_type=self.config["db_dense_index_type"],
                        metric_type=self.config["db_dense_metric_type"],
                        params=self.config["db_dense_params"]
                    )
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            db_logger.info(f"collection: {collection_name} 的索引创建成功")
        except Exception as e:
            db_logger.error(f"创建索引失败: {e}")
            raise

    # 插入数据
    def insert_data(self, collection_name: str, data: list[dict], embeddings: list[dict]):
        data = self._insert_data_preprocess(data, embeddings)
        try:
            res = self.client.insert(collection_name=collection_name, data=data)
            db_logger.info(f"表 {collection_name} 插入数据: {res['insert_count']} 条")
        except Exception as e:
            db_logger.error(f"插入数据失败: {e}")
            raise

    # 插入前encode数据
    def _insert_data_preprocess(self, data: list[dict], embeddings: list[dict]):
        for i, item in enumerate(data):
            item["dense_vector"] = embeddings[i]["dense"][0]
            item["sparse_vector"] = embeddings[i]["sparse"]
            # item["colbert_vector"] = embeddings[i]["colbert_vecs"]
            if "id" in item:
                del item["id"]
        return data

    # 搜索
    def search(
        self,
        collection_name: str,
        query_embedding: list,
        search_params: dict,
        limit: int,
        output_fields: list,
        search_field: str,
    ):
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                output_fields=output_fields,
                anns_field=search_field,
                search_params=search_params,
                limit=limit,
            )
            return results[0]
        except Exception as e:
            db_logger.error(f"搜索失败: {e}")
