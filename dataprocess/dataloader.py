import json
from rag.config.config import Config
import os
import requests
import jieba  
import re

from bs4 import BeautifulSoup
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer

def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_stopwords_form_txt(file_path):
    with open(file_path, encoding='utf-8') as f:
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
    if dataset_name == 'corpus':
        dataset_path = config['corpus_path']
    elif dataset_name == 'qa':
        dataset_path = config['qa_path']
    
    if not dataset_path:
        raise ValueError(f"未找到名为 {dataset_name} 的数据集路径。")
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, dataset_path)
    data_set = load_data_from_json(file_path)
    
    return data_set

def get_datasets():
    dataset_names = ["corpus", "qa"]
    ret = []
    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        ret.append(dataset)
    return ret

def load_stopwords():
    config = Config()
    
    stopword_path = config['stopword_path']
    current_dir = os.getcwd()
 
    file_path = os.path.join(current_dir, stopword_path)
    stopwords = load_stopwords_form_txt(file_path)
    return stopwords

def get_dict_path():
    config = Config()
    
    dict_path = config['dict_path']
    current_dir = os.getcwd()
    dict_path = os.path.join(current_dir, dict_path)
    
    return dict_path


def get_keywords(query):
    dict_path = get_dict_path()
    jieba.load_userdict(dict_path)
    new_data = ''.join(re.findall('[\u4e00-\u9fa5]+', query))
    stopwords = load_stopwords()
    
    seg_list = jieba.lcut(new_data)
    # 去除停用词并且去除单字
    filtered_words = [word for word in seg_list if word not in stopwords and len(word) > 1]
    # 使用TF-IDF对关键词进行打分
    tfidf_vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(filtered_words)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    keywords_with_weights = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    keywords_with_weights.sort(key=lambda x: x[1], reverse=True)  
    
    return keywords_with_weights



def get_html(query=None, url=None, mode='search'):
    """
    获取网页内容。
    
    参数：
        query: 搜索关键字(可选)
        url: 搜索网址(可选)
        mode: 工作模式(search: 搜索; extract: 提取)
    
    返回值：
        html_content: 网页内容
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
    }
    if mode == 'search' and query is not None:
        url = f"https://www.baidu.com/s?wd={query}"
    elif mode == 'extract' and url is not None:
        url = url

    if url is not None:
        try:
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'  # 设置编码格式
            
            if response.status_code == 200:
                html_content = response.text
                return html_content
            else:
                # print(f"Error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            # print(f"Error: {e}")
            return None
    else:
        print("Error: No URL provided!")
        return None


def parse_html(html_content, mode='search'):
    """
    解析网页内容。
    
    参数：
        html_content: 网页内容
        mode: 工作模式(search: 搜索; extract: 提取)
    
    返回值：
        results: 解析结果
    """
    if html_content:
        soup = BeautifulSoup(html_content, 'lxml')
    else:
        # print("Error: HTML content is empty!")
        return None
    if mode == 'search':
        results = []
        for item in soup.find_all('div', class_='result'):
            title = item.find('h3').get_text()
            link = item.find('a')['href']
            results.append({'title': title, 'link': link})
    elif mode == 'extract':
        doc = Document(html_content)
        content = doc.summary()
        soup_content = BeautifulSoup(content, 'lxml')
        content = soup_content.get_text(strip=True)
        # 未提取到内容则返回None
        if any('\u4e00' <= char <= '\u9fff' for char in content):
            results = content
        else:
            results = None
    return results


def content_summary(content,language='chinese'):
    """
    生成内容摘要。
    
    参数:
        content: 内容
        language: 语言类型

    返回值:
        summary: 内容摘要
    """
    if not content or len(content.strip()) == 0:
        print("输入内容为空")
    sentence_count = 5
    parser = PlaintextParser.from_string(content, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    
    summary = ""
    try:
        for sentence in summarizer(parser.document, sentence_count):
            summary += str(sentence) + " "
    except Exception as e:
        return f"摘要生成失败: {str(e)}"
    return summary

def baidu_search(query:str) -> list:
    search_content = get_html(query=query,mode='search')
    search_results = parse_html(search_content,'search')
    if search_results:
        return search_results
    return None
    

def bing_search(query: str) -> list:
    """
    使用 Bing Search API 搜索查询并获取结果。
    
    参数：
        query: 要搜索的查询字符串

    返回值：
        results: 搜索结果列表
    """
    config = Config()
    url = config['bing_url']
    api_key = config['bing_api']
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json"
    }
    params = {
        "q": query,
        "textDecorations": True,
        "textFormat": "HTML",
        "mtk": "zh-CN"  # 仅在中国市场搜索
    }
    response = requests.get(url, headers=headers, params=params)
   
    if response.status_code == 200:
        search_results = response.json()
        results = []
        
        if "webPages" in search_results:
            for item in search_results["webPages"]["value"]:
                results.append({
                    "title": item["name"],
                    "link": item["url"]
                })
        return results
    else:
        print(f"Error: {response.status_code}")
        return None
    
def get_docs(search_results:list):
    """
    获取检索结果中的文档内容。
    
    参数:
        search_results (list): 检索结果列表

    返回值:
        retrieved_docs: 文档内容列表
    """
    retrieved_docs = []
    max_len = 512
    if search_results:
        for result in search_results:
            url = result['link']
            title = result['title']
            doc_content = get_html(url=url,mode='extract')
            doc = parse_html(doc_content,'extract')
            if doc:
                doc = title + str(":") + str(doc)
                doc = content_summary(doc,'chinese')
                if len(doc) > max_len:
                    doc = doc[:max_len]
                elif len(doc) == 0:
                    continue
                retrieved_docs.append(doc)
            else:
                continue
            
    return retrieved_docs
        