import os
import json
import sys

# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.database.milvus import MilvusClientWrapper
from rag.config.config import Config
from rag.database.chunk import chunk_data
from pymilvus import model
from tqdm import tqdm
import torch

# config
config = Config()
# 设置环境变量以避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


path_regu = config["corpus_path"]
path_qa = config["qa_path"]

with open(path_regu, "r", encoding="utf-8") as file:
    data_regulation = json.loads(file.read())

with open(path_qa, "r", encoding="utf-8") as file:
    data_qa = json.loads(file.read())


wrapper = MilvusClientWrapper(config)

# 创建表，
wrapper.create_collection("regulation", data_regulation[0])
wrapper.create_collection("qa", data_qa[0])

# 索引创建
wrapper.set_index("regulation")
wrapper.set_index("qa")

# 分割text
chunked_data_regulation = chunk_data(data_regulation, config)
chunked_data_qa = chunk_data(data_qa, config)

embedding_model = model.hybrid.BGEM3EmbeddingFunction(
    device=config["embedding_device"], return_sparse=True, return_dense=True, return_colbert_vecs=True
)

def insert(chunked_data, collection_name, embedding_model):
    from tqdm import tqdm

    # 插入数据并显示进度条
    batch_size = 10
    total_batches = len(chunked_data) // batch_size + (1 if len(chunked_data) % batch_size != 0 else 0)
    for i in tqdm(range(total_batches), desc="Inserting data"):
        batch_data = chunked_data[i * batch_size : (i + 1) * batch_size]
        embeddings = [embedding_model.encode_queries([item["content"]]) for item in batch_data]
        wrapper.insert_data(collection_name, batch_data, embeddings)


insert(chunked_data_regulation, "regulation", embedding_model)
insert(chunked_data_qa, "qa", embedding_model)
