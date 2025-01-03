from typing import List

__all__ = ["build_prompt"]

def _build_fqa_promt(query_list: List[str], reranked_list: List[List[str]]) -> str:
    prompt = ""
    for i, (query, refs) in enumerate(zip(query_list, reranked_list), 1):
        prompt += f"问题: {query}\n 相关的FQA:\n"
        for ref in refs:
            prompt += f"- {ref}\n"
    return prompt

def _build_refer_prompt(query_list: List[str], reranked_list: List[List[str]]) -> str:
    prompt = ""
    for i, (query, refs) in enumerate(zip(query_list, reranked_list), 1):
        prompt += f"问题: {query}\n 参考文档:\n"
        for ref in refs:
            prompt += f"- {ref}\n"
    return prompt

# 暂时没用
def _build_web_search_prompt(query_list: List[str], reranked_list: List[List[str]]) -> str:
    prompt = ""
    for i, (query, refs) in enumerate(zip(query_list, reranked_list), 1):
        prompt += f"问题: {query}\n 搜索结果:\n"
        for ref in refs:
            prompt += f"- {ref}\n"
    return prompt

def build_prompt(query_list: List[str], fqa_reranked_list: List[List[str]], refer_rereanked_list: List[List[str]]) -> str:
    prompt = "你是一个海关问答助手，需要你根据参考文档回答问题：\n"
    prompt += _build_fqa_promt(query_list, fqa_reranked_list)
    prompt += _build_refer_prompt(query_list, refer_rereanked_list)
    prompt += "\n答案是："
    return prompt