import re

from rag.config.config import Config
from rag.generator.generator import Generator
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
    final_sentences = [sentence[:40] for sentence in cleaned_sentences]
    return final_sentences


text_list = split_text_into_sentences("data/corpus/test.txt")
query = ["中国特色社会主义有什么优点？", "中国特色社会主义主要体现在哪些方面？"]

config = Config()
model_path = 'model/BAAI/bge-base-zh-v1.5'
retriever1 = BM25Retriever(config)
retriever2 = DenseRetriever(config, None, model_path)

retriever1_result = retriever1.query(query, text_list)
retriever2_result = retriever2.query(query, text_list)
retriever_result = [a + b for a, b in zip(retriever1_result, retriever2_result)]
print(retriever_result)

model_path = 'model/BAAI/bge-reranker-v2-m3'
reranker = Reranker(config, model_path)
result = reranker.rerank(query, retriever_result)
print(result)

generator_model_name = 'model/Qwen/Qwen2.5-1.5B-Instruct'
generator = Generator(config, generator_model_name)
answer = generator.generate(query, text_list)
print(answer)

"""
输出：
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.514 seconds.
Prefix dict has been built successfully.
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
[['我们要坚持走中国特色社会主义法治道路 建设中国特色社会主义法治体系、建设社会主义', '——坚持中国特色社会主义道路', '实践告诉我们 中国共产党为什么能 中国特色社会主义为什么好 归根到底是马克思主义', '——我们全面加强党的领导 明确中国特色社会主义最本质的特征是中国共产党领导 中国', '必须坚定不移走中国特色社会主义政治发展道路 坚持党的领导、人民当家作主、依法治国', '一些人对中国特色社会主义政治制度自信不足 有法不依、执法不严等问题严重存在', '（二）坚持不懈用新时代中国特色社会主义思想凝心铸魂', '大会的主题是 高举中国特色社会主义伟大旗帜 全面贯彻新时代中国特色社会主义思想 ', '实践告诉我们 中国共产党为什么能 中国特色社会主义为什么好 归根到底是马克思主义', '中国共产党是为中国人民谋幸福、为中华民族谋复兴的党 也是为人类谋进步、为世界谋大', '我们要坚持走中国特色社会主义法治道路 建设中国特色社会主义法治体系、建设社会主义', '——坚持中国特色社会主义道路', '科学社会主义在二十一世纪的中国焕发出新的蓬勃生机 中国式现代化为人类实现现代化提', '——我们创立了新时代中国特色社会主义思想 明确坚持和发展中国特色社会主义的基本方', '（一）完善以宪法为核心的中国特色社会主义法律体系', '我国是一个发展中大国 仍处于社会主义初级阶段 正在经历广泛而深刻的社会变革 推进'], ['我们要坚持走中国特色社会主义法治道路 建设中国特色社会主义法治体系、建设社会主义', '——坚持中国特色社会主义道路', '实践告诉我们 中国共产党为什么能 中国特色社会主义为什么好 归根到底是马克思主义', '必须坚定不移走中国特色社会主义政治发展道路 坚持党的领导、人民当家作主、依法治国', '——我们全面加强党的领导 明确中国特色社会主义最本质的特征是中国共产党领导 中国', '一些人对中国特色社会主义政治制度自信不足 有法不依、执法不严等问题严重存在', '共同富裕是中国特色社会主义的本质要求 也是一个长期的历史过程', '大会的主题是 高举中国特色社会主义伟大旗帜 全面贯彻新时代中国特色社会主义思想 ', '实践告诉我们 中国共产党为什么能 中国特色社会主义为什么好 归根到底是马克思主义', '中国式现代化 是中国共产党领导的社会主义现代化 既有各国现代化的共同特征 更有基', '大会的主题是 高举中国特色社会主义伟大旗帜 全面贯彻新时代中国特色社会主义思想 ', '（一）完善以宪法为核心的中国特色社会主义法律体系', '全党必须牢记 坚持党的全面领导是坚持和发展中国特色社会主义的必由之路 中国特色社', '中国式现代化的本质要求是 坚持中国共产党领导 坚持中国特色社会主义 实现高质量发', '我们要坚持走中国特色社会主义法治道路 建设中国特色社会主义法治体系、建设社会主义', '我国是一个发展中大国 仍处于社会主义初级阶段 正在经历广泛而深刻的社会变革 推进']]
pre tokenize:   0%|          | 0/1 [00:00<?, ?it/s]You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 357.69it/s]
/home/ubuntu/anaconda3/envs/torch/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:2888: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
pre tokenize: 100%|██████████| 1/1 [00:00<00:00, 397.34it/s]
[['——我们全面加强党的领导 明确中国特色社会主义最本质的特征是中国共产党领导 中国', '实践告诉我们 中国共产党为什么能 中国特色社会主义为什么好 归根到底是马克思主义'], ['——我们全面加强党的领导 明确中国特色社会主义最本质的特征是中国共产党领导 中国', '必须坚定不移走中国特色社会主义政治发展道路 坚持党的领导、人民当家作主、依法治国']]
中国特色社会主义的优点包括：

1. 具有广泛的代表性，能够反映各个民族、各个阶层的利益。
2. 实行人民民主专政，保障了人民当家作主的权利。
3. 坚持党的领导，确保国家沿着正确的方向前进。
4. 推进改革开放和现代化建设，不断满足人民日益增长的美好生活需要。
5. 维护世界和平与发展，推动构建人类命运共同体。

综上所述，中国特色社会主义的优点在于其广泛代表性和坚持党的领导，以及推进改革开放和现代化建设等多方面的成就。
"""