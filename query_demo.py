from rag.config.config import Config
from rag.retriever.retriever import BM25Retriever, DenseRetriever

text_list = [
    "这是一段示例文本。",
    "这是另一段示例文本。",
    "这是第三段示例文本。",
    "这是第四段示例文本。",
    "这是第五段示例文本，",
    "123",
    "你好啊",
    "吃了吗",
    "我好",
]

query = ['你好，我想查询第一段']
config = Config()
model_path = 'model/BAAI/bge-base-zh-v1.5'
retriever1 = BM25Retriever(config)
retriever2 = DenseRetriever(config, None, model_path)

result1 = retriever1.query(query, text_list)
result2 = retriever2.query(query, text_list)

print(result1)
print(result2)
