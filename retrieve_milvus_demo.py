import time
import re
import json

from rag.milvus.milvus_client import CustomMilvusClient
from rag.retriever.retriever_mlivus import MilvusRetriever

# 时间上下文管理器
class TimeContextManager:
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"{self.description} cost time: {end_time - self.start_time} seconds")

# 分句子
def split_text_into_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    sentences = re.split(r'([。！？；])', content)
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        combined_sentences.append(sentences[i] + sentences[i + 1])
    if len(sentences) % 2 != 0:
        combined_sentences.append(sentences[-1])
    cleaned_sentences = [re.sub(r'[，。！？；：]', ' ', sentence).strip() for sentence in combined_sentences if
                         sentence.strip()]
    final_sentences = [sentence[:150] for sentence in cleaned_sentences]
    return final_sentences


# 数据读取
info_dict = {
    "document type":"", #法规类型
    "category":"", #法规类别
    "announcement_number":"", #文号
    "issuing authority":"", #发布机构
    "issue date":"", #发布日期
    "effective_date":"", #生效日期
    "status":"", #效力
    "remarks":"", #效力说明
    "title":"", #标题
    "content":"", #内容
    }

# 读取/data/regulations.json
# 构建JSON文件的相对路径
json_path = "data/regulations.json"
with open(json_path, 'r', encoding='utf-8') as f:
    regulations = json.load(f)

# TODO 语义分块,正从所看论文复现代码

# 初始化数据库
with TimeContextManager("MilvusClient init"):
    milvus_client = CustomMilvusClient()
    
# 创建表
with TimeContextManager("Create collection"):
    collection_name = "CustomRegulations"
    index_field_names = ["vector"]
    milvus_client.create_collection(collection_name,info_dict,index_field_names)

# 插入数据
with TimeContextManager("Insert data"):
    milvus_client.insert(collection_name, regulations[:20])

# 检索
query = "生物检疫"
collection_name = "CustomRegulations"
output_fields = [ "document_type", "category", "title", "content"]
milvusRetriever = MilvusRetriever(client=milvus_client.client, embedding_model=milvus_client.embedding_model)
res = milvusRetriever.denseQuery(collection_name = collection_name, query_list = query, output_fields = output_fields)
print(res)