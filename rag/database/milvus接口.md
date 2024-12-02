# database包含milvus.py，和utils.py

milvus.py包含两个类MilvusClientWrapper、MilvusService

utils.py提供了检索接口

**bgem3的colbert包含多向量，但milvus不支持存储多向量，也不支持检索多向量，所以暂时不支持colbert多向量检索**

## MilvusClientWrapper

MilvusClientWrapper是对milvus的python客户端的封装，提供了milvus的常用接口。

MilusCilentWrapper初始化后，会自动连接milvus服务，连接的地址为`milvus`环境变量指定的地址，如果没有指定，则默认为`localhost:19530`。

初始化

```python

client = MilvusClientWrapper(host,port)

参数：
host: str = 'localhost'
port: int = 19530

```


## MilvusService
服务类，目前实现了表的创建、数据插入、稠密检索、稀疏检索。后需完善增删改查

1. 初始化

```python

service = MilvusService(client, embedding_model)

参数：

client: MilvusClientWrapper
embedding_model: pymilvus封装的model。（尝试过用FlagEmbeddingModel，发现稀疏向量需要转换格式，比较麻烦，遂用pymilvus的model。pymilvus的model底层也是对FlagEmbeddingModel进行封装）

```

2. 创建表

```python

service.create_collection(collection_name,info_dict,index_field_names)

此函数创建表，同时会增加向量field, 命名为dense_vector sparse_vector，分别对应稠密向量和稀疏向量。

参数
collection_name: str 表名
info_dict:dict 表的field，如下，需至少包含“content”字段
    info_dict = {
        "document_type":"", #法规类型
        "category":"", #法规类别
        "announcement_number":"", #文号
        "issuing_authority":"", #发布机构
        "issue_date":"", #发布日期
        "effective_date":"", #生效日期
        "status":"", #效力
        "remarks":"", #效力说明
        "title":"", #标题
        "content":"", #内容
        }
index_field_names: list 索引的field，一般向量列,如index_field_names = ["dense_vector","sparse_vector"]

```

3. 插入数据

```python

service.insert_or_update(collection_name, data)

参数：
collection_name: str 表名
data:list[dict] 每个dict为一条数据，包含表的所有字段，如下:
    data = [
        {
            "document_type":"xxxx", #法规类型
            "category":"xxx", #法规类别
            "announcement_number":"xxxx", #文号
            "issuing_authority":"xxxx", #发布机构
            "issue_date":"xxxx", #发布日期
            "effective_date":"xxxx", #生效日期
            "status":"xxxx", #效力
            "remarks":"xxxx", #效力说明
            "title":"xxxx", #标题
            "content":"xxxx", #内容
        },
        ...
    ]

```

## utils.py

提供了检索接口
1. 稀疏检索

```python

utils.dense_search(service, collection_name, query_embedding:list, output_fields, topk=10)

参数
service: MilvusService
collection_name: str 表名
query_embedding: list 查询的向量表示
output_fields: list 输出info_dict里的字段列表
topk: int = 10 返回的结果数量

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

utils.dense_search(service, collection_name, query_embedding:list, output_fields, topk=10)

```

3. colbert多向量
   
milvus不支持存储colbert多向量，也不支持检索colbert多向量，所以暂时不支持colbert多向量检索
