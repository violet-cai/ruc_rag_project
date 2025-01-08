from openai import OpenAI
import json



def get_response(messages:list, llm_model:str):
    """
    调用LLM获取回复
    """
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.7
        )
        message_content = response.choices[0].message.content.strip()
        return message_content
    except Exception as e:
        return f"调用LLM分析内容时出现错误: {e}"
    

def get_summary(text:str,llm_model:str):
    """
    对历史内容进行总结得到摘要
    """
    messages=[
            {"role": "system", "content":'''The following text is a series of conversations between a user and a customs agent assistant. Please summarize the content of the conversation.
            Requirements: The summary should be as concise as possible, while retaining the user's demands and the solutions given by the assistant; in addition, the summary may be used for subsequent problem descriptions, so please pay attention to the accuracy of the summary.
            '''},
            {"role": "user", "content": text}
        ]
    summary = get_response(messages, llm_model)
    return summary


def get_new_query_1(historys:list, query:str):
    """
    拼接历史摘要 + 上轮对话 + 当前query得到新的query
    
    Attributes:
        historys (list): A list containing the previous conversations between the user and the assistant.
        query (str): The user's current question.
    Returns:
        new_query (str): A new, structured query that incorporates the context of previous interactions.
    """
    llm_model = 'qwen-max-latest'
    if historys:
        history = json.dumps(historys[:-1], ensure_ascii=False)
        summary = get_summary(history, llm_model)
        last_conversation = json.dumps(historys[-1])
        if last_conversation == None:
            last_conversation = ''
        total_content = f"summary:{summary}\nlast_conversation:{last_conversation}\nquery:{query}"
        messages=[
            {"role": "system", "content":'''The following text contains 3 parts: 1.summary: a summary of a series of conversations between the user and the customs assistant; 2.last_conversation: the last round of conversation between the user and the assistant; 3.query: the user's current question. Based on the above information, please make an accurate and comprehensive summary of the user's current question so that the assistant can better retrieve relevant information. Note: You only need to give the summaried questions.
            '''},
            {"role": "user", "content": total_content}
        ]
        new_query = get_response(messages, llm_model)
    else:
        new_query = query
    return new_query    
    
    
def get_new_query_2(historys:list, query:str):
    """
    拼接历史对话 + 当前query得到新的query
    
    Attributes:
        historys (list): A list containing the previous conversations between the user and the assistant.
        query (str): The user's current question.
    Returns:
        new_query (str): A new, structured query that incorporates the context of previous interactions.
    """
    llm_model = 'qwen-max-latest'
    if historys:
        history = json.dumps(historys, ensure_ascii=False)
        total_content = f"history:{history}\nquery:{query}"
        messages=[
            {"role": "system", "content":'''The following text contains 2 parts: 1.history: the historical conversations between the user and the assistant; 2.query: the user's current question. Based on the above information, please make an accurate and comprehensive summary of the user's current question so that the assistant can better retrieve relevant information. Note: You only need to give the summaried questions.
            '''},
            {"role": "user", "content": total_content}
        ]
        new_query = get_response(messages, llm_model)
    else:
        new_query = query
    return new_query


client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",api_key  = "sk-c02056e042844ee1815cb64d8ede554f")



