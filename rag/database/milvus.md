# database包含milvus.py，和utils.py

milvus.py主要是MilvusClientWrapper

utils.py提供了检索接口

**bgem3的colbert包含多向量，但milvus不支持存储多向量，也不支持检索多向量，所以暂时不支持colbert多向量检索**

## MilvusClientWrapper

MilvusClientWrapper是对milvus的python客户端的封装，提供了milvus的常用接口。
相关设置在basic_config.yaml中，如下：

本地数据库

```yaml
db_uri可为:
"http://localhost:19530" # docker
"./milvus.db" # 本地

db_embedding_dim: 1024 # 稠密向量维度设置,同使用的embedding维度一致
```

初始化

```python

client = MilvusClientWrapper(config)

参数：

```

## 功能接口

1. 创建表

```python

collection_name = "regulation"
# 创建表
wrapper.create_collection(collection_name, data_regulation[0])

此函数创建表，同时会增加两个向量字段, 命名为dense_vector sparse_vector，分别对应稠密向量和稀疏向量。

参数
collection_name: str 表名
data_regulation[0]:传入的表的第一条数据，用于获取表的字段名

```

basic_config.yaml中的参数

```yaml
# 一些重要字段的key,用于创建表创建
db_text_id: "text_id" 
db_content_key: "content"
db_appendix_content_key: "appendix_content"
db_metadata_key: "metadata"
```

3. 创建索引

```python
wrapper.set_index(collection_name)
参数为表名
```

```yaml
# 需要简历索引的字段
db_index_fields: ["dense_vector","sparse_vector"]
# 索引类型
db_sparse_index_type: "SPARSE_INVERTED_INDEX"
db_dense_index_type: "IVF_FLAT"
```

4. chunk.py

提供了chunk_data(data_regulation, config)
```python
chunk_data(data_regulation, config)

参数:
data_regulation: list[dict] 数据集,为dataset的数据
对content字段进行分块，返回新的数据集
```

```yaml
db_chunk_size: 256 # 分块大小,并非完全截断，而是截断到最近的一个句号
```

5. 插入数据

```python

wrapper.insert_data(collection_name, batch_data, embeddings)

参数：
collection_name: str 表名
data:list[dict] 每个dict为一条数据，包含表的所有字段，如下:
embeddings:list[dict] ,batch_data的每条数据,对content字段进行embedding,包含dense_vector和sparse_vector，如下:
[{
    dense:[dense_vector]
    sparse:sparse_vector
},
...
]

参考:
def insert(chunked_data, collection_name):
    from tqdm import tqdm
    # 插入数据并显示进度条
    batch_size = 100
    total_batches = len(chunked_data) // batch_size + (1 if len(chunked_data) % batch_size != 0 else 0)
    for i in tqdm(range(total_batches), desc="Inserting data"):
        batch_data = chunked_data[i * batch_size : (i + 1) * batch_size]
        embeddings = []
        for item in batch_data:
            embedding = {}
            dense_embedding = onnx_embedding_model.encode_documents([item["content"]])
            sparse_embedding = bge_embedding_model.encode_queries([item["content"]])
            embedding["dense"] = dense_embedding
            embedding["sparse"] = sparse_embedding["sparse"]
            # print(embedding)
            embeddings.append(embedding)
        wrapper.insert_data(collection_name, batch_data, embeddings)
        
insert(chunked_data_regulation, collection_name)
```

## utils.py

提供了检索接口
1. 稀疏检索

```python

utils.dense_search(query_embedding: list, topk=10):

参数
query_embedding: list 查询的向量表示

返回
type:list[dict],如下：
[{'id': 454187902015382266,
  'distance': 0.8868650794029236,
  'entity': {'title': '供港澳食用陆生动物检验检疫管理办法（海关总署第266号令）'}},
 {'id': 454187902015382588,
  'distance': 0.9028371572494507,
  'entity': {'title': '海关总署公告2022年第57号（关于发布《进境种用雏禽指定隔离检疫场建设规范》等90项行业标准的公告）'}}
  ...
]
distance为相似度，entity里为output_fields里的字段
```
2. 稀疏检索

同稠密向量检索，只是调用的是utils.sparse_search

```python

utils.sparse_search(query_embedding: list, topk=10):

```
