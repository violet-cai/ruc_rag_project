import importlib
from openai import OpenAI
from rag.config.config import Config
import json

config = Config()
client = OpenAI(
    base_url=config['base_url'], api_key=config['api_key']
)


def get_retriever(config):
    return getattr(importlib.import_module("rag.retriever"), "Retriever")(config)


def get_response(messages: list, llm_model: str):
    """
    调用LLM获取回复
    """
    try:
        response = client.chat.completions.create(model=llm_model, messages=messages, temperature=0.7)
        message_content = response.choices[0].message.content.strip()
        return message_content
    except Exception as e:
        return f"调用LLM分析内容时出现错误: {e}"


def get_summary(text: str, llm_model: str):
    """
    对历史内容进行总结得到摘要
    """
    prompt = config['summarize_text']
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": text},
    ]
    summary = get_response(messages, llm_model)
    return summary


def get_new_query_1(query: str, historys: list):
    """
    拼接历史摘要 + 上轮对话 + 当前query得到新的query

    参数:
        historys (list): 包含用户和助理之间之前对话的列表。
        query (str): 用户当前的问题。
    返回:
        new_query (str): 一个新的、结构化的查询，结合了之前交互的上下文。
    """
    llm_model = config['llm_model']
    prompt = config['update_query_1']
    if historys:
        history = json.dumps(historys[:-1], ensure_ascii=False)
        summary = get_summary(history, llm_model)
        last_conversation = json.dumps(historys[-1])
        if last_conversation == None:
            last_conversation = ""
        total_content = f"summary:{summary}\nlast_conversation:{last_conversation}\nquery:{query}"
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": total_content},
        ]
        new_query = get_response(messages, llm_model)
    else:
        new_query = query
    return new_query


def get_new_query_2(query: str, historys: list):
    """
    拼接历史对话 + 当前query得到新的query

    参数:
        historys (list): 包含用户和助理之间之前对话的列表。
        query (str): 用户当前的问题。
    返回:
        new_query (str): 一个新的、结构化的查询，结合了之前交互的上下文。
    """
    llm_model = config['llm_model']
    prompt = config['update_query_2']
    if historys:
        history = json.dumps(historys, ensure_ascii=False)
        total_content = f"history:{history}\nquery:{query}"
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": total_content},
        ]
        new_query = get_response(messages, llm_model)
    else:
        new_query = query
    return new_query

