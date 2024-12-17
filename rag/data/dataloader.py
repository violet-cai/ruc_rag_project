import json
from ..config.config import Config
import os

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
    if dataset_name == 'dataset':
        dataset_path = config['corpus_path']
    elif dataset_name == 'qa':
        dataset_path = config['qa_path']
        
    if not dataset_path:
        raise ValueError(f"未找到名为 {dataset_name} 的数据集路径。")
    current_dir = os.path.dirname(__file__)
    if 'data/' in dataset_path:
       dataset_path = dataset_path.replace('data/', '') 
    file_path = os.path.join(current_dir, dataset_path)
    data_set = load_data_from_json(file_path)
    
    return data_set


def load_stopwords():
    config = Config()
    
    stopword_path = config['stopword_path']
    current_dir = os.path.dirname(__file__)
    if 'data/' in stopword_path:
       stopword_path = stopword_path.replace('data/', '') 
    file_path = os.path.join(current_dir, stopword_path)
    stopwords = load_stopwords_form_txt(file_path)
    return stopwords

def get_dict_path():
    config = Config()
    
    dict_path = config['dict_path']
    current_dir = os.path.dirname(__file__)
    if 'data/' in dict_path:
       dict_path = dict_path.replace('data/', '') 
    dict_path = os.path.join(current_dir, dict_path)
    
    return dict_path