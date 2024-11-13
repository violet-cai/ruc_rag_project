import json
import re

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

json_path = "data/corpus/regulations.json"
with open(json_path, 'r', encoding='utf-8') as f:
    regulations = json.load(f)

info_dict = {
    "document type": "",  # 法规类型
    "category": "",  # 法规类别
    "announcement_number": "",  # 文号
    "issuing authority": "",  # 发布机构
    "issue date": "",  # 发布日期
    "effective_date": "",  # 生效日期
    "status": "",  # 效力
    "remarks": "",  # 效力说明
    "title": "",  # 标题
    "content": "",  # 内容
    "appendix":""  #附件
}

sorted_regulations = sorted(regulations, key=lambda x: len(x["content"]), reverse=True)
for i in range(10):
    single_sentences_list = re.split(r'[。；？！]+', sorted_regulations[i]["content"])
    print("___________________________________________________________________________")
    for j in range(len(single_sentences_list)):
        print(single_sentences_list[j])

# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)
#
#
# def calculate_cosine_distances(sentences):
#     distances = []
#     for i in range(len(sentences) - 1):
#         embedding_current = sentences[i]['combined_sentence_embedding']
#         embedding_next = sentences[i + 1]['combined_sentence_embedding']
#         similarity = cosine_similarity(embedding_current, embedding_next)
#         distance = 1 - similarity
#         distances.append(distance)
#         sentences[i]['distance_to_next'] = distance
#     return distances, sentences
#
#
# texts = [regulations[i]["title"] + "\n" + regulations[i]["content"] for i in range(len(regulations))]
# text = texts[0]
# single_sentences_list = re.split(r'[。；？！\n]+', text)
# sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
#
# buffer_size = 1
# combined_sentences = [
#     ' '.join(
#         sentences[j]['sentence'] for j in range(max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))))
#     for i in range(len(sentences))
# ]
# for i, combined_sentence in enumerate(combined_sentences):
#     sentences[i]['combined_sentence'] = combined_sentence
# model = SentenceTransformer("model/BAAI/bge-base-zh-v1.5")
# embeddings = model.encode([x['combined_sentence'] for x in sentences])
# for i, sentence in enumerate(sentences):
#     sentence['combined_sentence_embedding'] = embeddings[i]
# distances, sentences = calculate_cosine_distances(sentences)
# plt.plot(distances)
# plt.show()
