from rag.config.config import Config
from rag.retriever.retriever import BM25Retriever, DenseRetriever, Reranker

text_list = [
    "小猫很可爱",
    "数学课真的很无聊啊",
    "智能体系结构也很不错",
    "赶紧做完这个项目就去睡觉",
    "蛋糕的基础材料是面粉、糖、鸡蛋和黄油",
    "编程语言的选择取决于项目需求",
    "制作蛋糕的第一步是混合干性成分",
    "Python 是一种广泛使用的高级编程语言"
]

query = [
    "如何制作蛋糕",
    "最好的编程语言是什么",
    "你喜欢什么动物",
    "什么时候睡觉",
]
config = Config()
model_path = 'model/BAAI/bge-base-zh-v1.5'
retriever1 = BM25Retriever(config)
retriever2 = DenseRetriever(config, None, model_path)

retriever1_result = retriever1.query(query, text_list)
retriever2_result = retriever2.query(query, text_list)

print(retriever1_result)
print(retriever2_result)

model_path = 'model/BAAI/bge-reranker-v2-m3'
reranker = Reranker(config, model_path)
result1 = reranker.rerank(query, retriever1_result)
result2 = reranker.rerank(query, retriever2_result)
print(result1)
print(result2)

"""
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.314 seconds.
Prefix dict has been built successfully.
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
[['制作蛋糕的第一步是混合干性成分', '蛋糕的基础材料是面粉、糖、鸡蛋和黄油', '小猫很可爱'], ['Python 是一种广泛使用的高级编程语言', '编程语言的选择取决于项目需求', '制作蛋糕的第一步是混合干性成分'], ['小猫很可爱', '数学课真的很无聊啊', '智能体系结构也很不错'], ['赶紧做完这个项目就去睡觉', '小猫很可爱', '数学课真的很无聊啊']]
[['蛋糕的基础材料是面粉、糖、鸡蛋和黄油', '制作蛋糕的第一步是混合干性成分', '赶紧做完这个项目就去睡觉'], ['Python 是一种广泛使用的高级编程语言', '编程语言的选择取决于项目需求', '智能体系结构也很不错'], ['小猫很可爱', 'Python 是一种广泛使用的高级编程语言', '编程语言的选择取决于项目需求'], ['赶紧做完这个项目就去睡觉', '数学课真的很无聊啊', '小猫很可爱']]
pre tokenize:   0%|          | 0/1 [00:00<?, ?it/s]You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 861.08it/s]
/home/ubuntu/anaconda3/envs/torch/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:2888: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 1057.83it/s]
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 964.21it/s]
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 1029.28it/s]
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 929.38it/s]
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 824.19it/s]
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 914.19it/s]
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 1106.68it/s]
[['制作蛋糕的第一步是混合干性成分'], ['Python 是一种广泛使用的高级编程语言'], ['小猫很可爱'], ['赶紧做完这个项目就去睡觉']]
[['制作蛋糕的第一步是混合干性成分'], ['Python 是一种广泛使用的高级编程语言'], ['小猫很可爱'], ['赶紧做完这个项目就去睡觉']]
"""