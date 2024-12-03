import re
from concurrent.futures import ThreadPoolExecutor


def fixed_chunk(text: str, chunk_size: int) -> list[str]:
    """
    将文本分割成固定大小的块，避免切断句子

    参数:
    text (str): 要分割的文本
    chunk_size (int): 每个块的大小

    返回:
    list: 分割后的文本块列表
    """
    sentences = re.split(r'(?<=[。！？])', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def process_document(doc, chunk_size):
    title = doc["title"]
    chunk_list = fixed_chunk(doc["content"], chunk_size)
    chunked_docs = []
    for chunk_text in chunk_list:
        chunk = dict(doc)
        chunk["content"] = title + ":" + chunk_text
        chunked_docs.append(chunk)
    return chunked_docs

def chunk_data(data: list[dict], config: dict) -> list[dict]:
    """
    将.json文件中的数据分块
    
    参数:
    data (list[dict]): .json文件中的数据
    config (dict): 配置文件
    
    返回:
    list[dict]: 分块后的数据,同data数据结构
    """
    chunk_size = config["db_chunk_size"]
    chunk_data = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, doc, chunk_size) for doc in data]
        for future in futures:
            chunk_data.extend(future.result())

    return chunk_data