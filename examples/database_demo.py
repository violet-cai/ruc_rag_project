import os
import json
import sys

# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.database.milvus import MilvusClientWrapper, MilvusService
from rag.config.config import Config
from rag.database.chunk import chunk_data
from pymilvus import model
from tqdm import tqdm


# config
config = Config()
# 设置环境变量以避免内存碎片化
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.cuda.empty_cache()  # 清空CUDA缓存
# 初始化客户端和服务端
client = MilvusClientWrapper()
# embedding_model = BGEM3FlagModel(config["embedding_model"])
# embedding_model = ""
embedding_model = model.hybrid.BGEM3EmbeddingFunction(
    device="cpu", use_fp16=False, return_sparse=True, return_dense=True, return_colbert_vecs=False
)
service = MilvusService(client, embedding_model)

# 表结构
info_dict = {
    "document_type": "",  # 法规类型
    "category": "",  # 法规类别
    "announcement_number": "",  # 文号
    "issuing_authority": "",  # 发布机构
    "issue_date": "",  # 发布日期
    "effective_date": "",  # 生效日期
    "status": "",  # 效力
    "remarks": "",  # 效力说明
    "title": "",  # 标题
    "content": "",  # 内容
}

# 创建表
collection_name = config["db_collection_name"]
# milvus数据库好像不能存储colbert_vecs，只能存储dense_vector和sparse_vector
# 先实现dense_vector和sparse_vector的存储和检索
index_field_names = config["db_index_fields"]
service.create_collection(collection_name, info_dict, index_field_names)

# 构建JSON文件的相对路径
json_path = "data/regulations_with_metadata.json"
# 读取JSON文件
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 分割text
chunked_data = chunk_data(data, config)

# 插入数据
# 100 为例
chunked_data = chunked_data[:100]
# 插入数据并显示进度条
batch_size = 10
total_batches = len(chunked_data) // batch_size + (1 if len(chunked_data) % batch_size != 0 else 0)

for i in tqdm(range(total_batches), desc="Inserting data"):
    batch_data = chunked_data[i * batch_size : (i + 1) * batch_size]
