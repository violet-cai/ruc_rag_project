import json
from ..config.config import Config
import os
import requests
import jieba
import re
from bs4 import BeautifulSoup
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_stopwords_form_txt(file_path):
    with open(file_path, encoding="utf-8") as f:
        stopwords = {line.strip() for line in f.readlines()}
    return stopwords


def load_dataset(dataset_name):
    """
    根据数据集名称加载相应的数据集.

    参数：
    dataset_name (str): 数据集名称。

    取值：
    dataset: regulations数据集;
    qa: qa数据集。

    返回：
    dict: 加载的数据集
    """
    config = Config()
    if dataset_name == "dataset":
        dataset_path = config["corpus_path"]
    elif dataset_name == "qa":
        dataset_path = config["qa_path"]

    if not dataset_path:
        raise ValueError(f"未找到名为 {dataset_name} 的数据集路径。")
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "..", dataset_path)
    file_path = os.path.normpath(file_path)
    data_set = load_data_from_json(file_path)

    return data_set


def load_stopwords():
    config = Config()

    stopword_path = config["stopword_path"]
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "..", stopword_path)
    file_path = os.path.normpath(file_path)
    stopwords = load_stopwords_form_txt(file_path)
    return stopwords


def get_dict_path():
    config = Config()

    dict_path = config["dict_path"]
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "..", dict_path)
    return file_path


def get_keywords(query):
    dict_path = get_dict_path()
    jieba.load_userdict(dict_path)
    new_data = "".join(re.findall("[\u4e00-\u9fa5]+", query))
    stopwords = load_stopwords()

    seg_list = jieba.lcut(new_data)
    # 去除停用词并且去除单字
    filtered_words = [word for word in seg_list if word not in stopwords and len(word) > 1]
    # 使用TF-IDF对关键词进行打分
    tfidf_vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(filtered_words)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    keywords_with_weights = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    keywords_with_weights.sort(key=lambda x: x[1], reverse=True)

    return keywords_with_weights


def get_html(query=None, url=None, mode="search"):
    """
    获取网页内容。

    参数：
    query: 搜索关键字(可选)
    url: 搜索网址(可选)
    mode: 工作模式(search: 搜索; extract: 提取)

    返回：
    html_content: 网页内容
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36"
    }
    if mode == "search" and query is not None:
        url = f"https://www.baidu.com/s?wd={query}"
    elif mode == "extract" and url is not None:
        url = url

    if url is not None:
        response = requests.get(url, headers=headers)
        response.encoding = "utf-8"  # 设置编码格式
        if response.status_code == 200:
            html_content = response.text
            return html_content
        else:
            print("Error: Failed to retrieve HTML content!")
            return None
    else:
        print("Error: No URL provided!")
        return None


def parse_html(html_content, mode="search"):
    """
    解析网页内容。

    参数：
    html_content: 网页内容
    mode: 工作模式(search: 搜索; extract: 提取)

    返回：
    results: 解析结果
    """
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
    else:
        # print("Error: HTML content is empty!")
        return None
    if mode == "search":
        results = []
        for item in soup.find_all("div", class_="result"):
            title = item.find("h3").get_text()
            link = item.find("a")["href"]
            results.append({"title": title, "link": link})
    elif mode == "extract":
        doc = Document(html_content)
        content = doc.summary()
        soup_content = BeautifulSoup(content, "html.parser")
        content = soup_content.get_text(strip=True)
        # 未提取到内容则返回None
        if any("\u4e00" <= char <= "\u9fff" for char in content):
            results = content
        else:
            results = None
    return results
