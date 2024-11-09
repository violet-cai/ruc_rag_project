import re
import time
import concurrent.futures

from rag.config.config import Config
import rag.generator.generator
from rag.retriever.retriever import BM25Retriever, DenseRetriever, Reranker


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


text_list = split_text_into_sentences("data/corpus/test.txt")
total = time.time()
query = "中国特色社会主义有什么优点，我们为什么要坚持走特色社会主义道路，特色社会主义道路带给我们什么"
# TODO 完成对query进行修改得到query_list
query_list = ["中国特色社会主义的优点是什么？", "我们为什么要坚持走中国特色社会主义道路？",
              "中国特色社会主义道路带给我们什么？"]

# TODO 对比实验看一下retrieval应该怎么分配权重
t = time.time()
config = Config()
model_path = 'model/BAAI/bge-base-zh-v1.5'
retriever1 = BM25Retriever(config)
# TODO 这里除了异步还有什么加速方法可以找一下，query_instruction看看怎么使用
retriever2 = DenseRetriever(config, None, model_path)
with concurrent.futures.ThreadPoolExecutor() as executor:
    future1 = executor.submit(retriever1.query, query_list, text_list)
    future2 = executor.submit(retriever2.query, query_list, text_list)
    retriever1_result = future1.result()
    retriever2_result = future2.result()
retriever_result = [a + b for a, b in zip(retriever1_result, retriever2_result)]
print(retriever_result)
print(f'retrieve:{time.time() - t:.4f}s')

t = time.time()
# TODO 时间太长不知道是不是设备性能问题，还是rerank没写好，比tmd生成回答还蛮，需要优化！
model_path = 'model/BAAI/bge-reranker-v2-m3'
reranker = Reranker(config, model_path)
result = reranker.rerank(query_list, retriever_result)
print(result)
print(f'rerank:{time.time() - t:.4f}s')

t = time.time()
rag.generator.generator.generate(query_list=query_list, retrieved_list=result)
print(f'generate:{time.time() - t:.4f}s')
print(f'total:{time.time() - total:.4f}s')
