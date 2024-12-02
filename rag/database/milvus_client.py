from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from rag.database.logger import db_logger
from pymilvus import model


class CustomMilvusClient:
    def __init__(self, host="localhost", port="19530"):
        self.client = MilvusClient(host=host, port=port)
        self.embedding_model = self._enable_embedding()
        # self.vector_dim = self.embedding_model.dim()["dense"]
        self.vector_dim = 1024
        db_logger.info(f"Milvus 连接到 {host}:{port}")

    # 创建collection
    # fields: schema列表
    # 输入collection_name:str
    # schema_fields:[schema1,schema2,...]
    # index_field_names:[str1,str2,...]
    def create_collection(self, collection_name, info_dict, index_field_names):
        schema = self._create_milvus_schema(info_dict, self.vector_dim)
        # 索引
        index_params = self.client.prepare_index_params()
        for field_name in index_field_names:
            index_params.add_index(
                field_name=field_name, index_type="IVF_FLAT", metric_type="L2", params={"nlist": 1024}
            )
        try:
            # 创建collection
            if self.client.has_collection(collection_name):
                db_logger.info(f"collection: {collection_name} 已存在")
            else:
                res = self.client.create_collection(
                    collection_name=collection_name, name=collection_name, schema=schema, index_params=index_params
                )
                db_logger.info(f"创建collection: {collection_name} 返回结果: {res}")
        except Exception as e:
            db_logger.error(f"创建collection: {collection_name} 失败, 错误信息: {e}")

    def _create_milvus_schema(self, info_dict: dict, dim: int) -> CollectionSchema:
        fields = []
        # info_dict没有id字段，需要添加
        if "ID" not in info_dict.keys():
            info_dict["ID"] = ""
        for key, value in info_dict.items():
            if key == "ID":
                fields.append(FieldSchema(name=key, dtype=DataType.INT64, is_primary=True, auto_id=True))
            elif key == "content":
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=65535, is_primary=False))
            else:
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=255, is_primary=False))
        # 创建一个向量模型输出维度的浮点向量
        vec_col = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="向量",
        )
        fields.append(vec_col)
        schema = CollectionSchema(fields=fields, description="Schema for info_dict")
        return schema

    # 选择embedding 模型
    # 此处默认使用BGEM3EmbeddingFunction，后续按需求增加新模型
    def _enable_embedding(self):
        embedding_fn = model.hybrid.BGEM3EmbeddingFunction()
        db_logger.info(f"embedding模型: {embedding_fn}")
        return embedding_fn

    # 插入向量
    # 插入数据
    # collection_name: str
    # chunk_list: list[dict]
    def insert(self, collection_name, chunk_list: list[dict]):
        embedding_fn = self.embedding_model
        # 获取集合
        client = self.client
        # 提取嵌入
        embeddings = [embedding_fn([item["content"]])["dense"][0] for item in chunk_list]
        # 构建插入数据
        data = chunk_list
        # 增加向量列
        for i, item in enumerate(data):
            item["vector"] = embeddings[i]
        # 插入数据
        res = self.client.insert(collection_name=collection_name, data=data)
        db_logger.info(f"表 {collection_name} 插入数据: {res['insert_count']} 条")

    def query(self, query_embeddings, collection_name=None, top_k=5):
        # 后续修改这个接口，我随便改的
        try:
            res = self.client.search(
                collection_name=collection_name,
                data=query_embeddings["dense"],
                search_params={"metric_type": "L2"},
                limit=top_k,
            )
        except Exception as e:
            db_logger.error(f"检索失败, 错误信息: {e}")
        res_list = []
        return res_list

    # 选择embedding 模型
    # 此处默认使用BGEM3EmbeddingFunction，后续按需求增加新模型
    def _enable_embedding(self):
        embedding_fn = model.hybrid.BGEM3EmbeddingFunction()
        db_logger.info(f"embedding模型: {embedding_fn}")
        return embedding_fn

    # 将res的title content字段连接为字符串：
    def _res_to_string(self, res, output_fields) -> str:
        knowledge = ""
        for i, res_i in enumerate(res):
            knowledge += "====================\n"
            knowledge += f"The response of {i + 1} query:\n"
            for j, row in enumerate(res_i):
                knowledge += f"\tTop {j + 1}:\n"
                for field in output_fields:
                    knowledge += f"\t{field}: {row['entity'][field]}\n"
        return knowledge
