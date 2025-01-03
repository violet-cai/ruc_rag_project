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
from pymilvus import model, RRFRanker,AnnSearchRequest
from tqdm import tqdm

# config
config = Config()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

path_regu = config["corpus_path"]
path_qa = config["qa_path"]

with open(path_regu, "r", encoding='utf-8') as file:
	data_regulation = json.loads(file.read())
 
with open(path_qa, "r", encoding='utf-8') as file:
	data_qa = json.loads(file.read())

# 去除没有metadata的数据(有数据没有metadata)
# data_regulation = [data for data in data_regulation if data.get('metadata')]
data_qa = [data for data in data_qa if data.get('metadata')]
# 将data_qa的"content"加上"answer"字段
for data in data_qa:
    data["content"] = "question:\n" +data["content"] + "\nanswer:\n" +data["answer"]

# # 统计data_qa中data['answer']的最大长度
# max_length = max([len(data['content']) for data in data_qa])
# print(max_length)


# embedding_model = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, return_dense=True, return_colbert_vecs=True)
wrapper = MilvusClientWrapper(config)
# collection_name = "regulation"
collection_name = "qa"
# 创建表
# wrapper.create_collection(collection_name, data_regulation[0])
wrapper.create_collection("qa", data_qa[0])
# 索引创建
# wrapper.set_index(collection_name)
wrapper.set_index('qa')

from rag.database.chunk import chunk_data
# 分割text,qa不分割
# chunked_data_regulation = chunk_data(data_regulation, config)
chunked_data_qa = data_qa

# 插入数据
# 100 为例
# chunked_data_regulation = chunked_data_regulation[:100]
# chunked_data_qa = chunked_data_qa[:100]
# bge_embedding_model = model.hybrid.BGEM3EmbeddingFunction(device='cuda:0', return_sparse=True, return_dense=False, return_colbert_vecs=False)
# onnx_embedding_model = model.DefaultEmbeddingFunction()
bge_embedding_model = model.hybrid.BGEM3EmbeddingFunction(device='cuda:0', return_sparse=True, return_dense=True, return_colbert_vecs=False)

def insert(chunked_data, collection_name):
    from tqdm import tqdm
    # 插入数据并显示进度条
    batch_size = 10
    total_batches = len(chunked_data) // batch_size + (1 if len(chunked_data) % batch_size != 0 else 0)
    for i in tqdm(range(total_batches), desc="Inserting data"):
        batch_data = chunked_data[i * batch_size : (i + 1) * batch_size]
        embeddings = []
        for item in batch_data:
            # embedding = {}
            # dense_embedding = onnx_embedding_model.encode_documents([item["content"]])
            # sparse_embedding = bge_embedding_model.encode_queries([item["content"]])
            # embedding["dense"] = dense_embedding
            # embedding["sparse"] = sparse_embedding["sparse"]
            # print(embedding)
            # embeddings.append(embedding)
            embeddings = [bge_embedding_model.encode_queries([item["content"]]) for item in batch_data]
        wrapper.insert_data(collection_name, batch_data, embeddings)
        
insert(chunked_data_qa, collection_name)
# # insert(chunked_data_qa, "qa", embedding_model)

# query = "海关条例"
# collection_name = "regulation"
# k = 5
# full_text_search_params = {"metric_type": "BM25"}
# dense = onnx_embedding_model.encode_queries([query])
# dense_vec = dense[0]
# output_fields = ["content"]
# full_text_search_req = AnnSearchRequest(
#     [query], "bm25_sparse_vector", full_text_search_params, limit=k
# )

# dense_search_params = {"metric_type": "IP"}
# dense_req = AnnSearchRequest(
#     [dense_vec], "dense_vector", dense_search_params, limit=k
# )

# results = wrapper.client.hybrid_search(
#     collection_name,
#     [full_text_search_req, dense_req],
#     ranker=RRFRanker(),
#     limit=k,
#     output_fields=output_fields,
# )

# print(results)