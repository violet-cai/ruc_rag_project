from collections import defaultdict
import concurrent
import json
from openai import OpenAI
import os


def get_response(text,llm_model):
    """
    使用LLM对内容进行分析
    """
    messages=[
            {"role": "system", "content":'''You are an excellent text summary assistant. The following text is the title, content, question type and answer of a question involving customs regulations. Please analyze the content of the text to determine whether the text involves specific laws and regulations or customs announcements, and return it in the form of a dictionary. The format is as follows：
                {
                "flag": boolean,(Return true if involved, otherwise return false)
                "laws":["xxx",.....](The format of the returned specified name should be complete, for example, the book title quotation marks and the corresponding annotation brackets should be returned together.)
                }
                (Important note: Please do not give anything other than the required dictionary, otherwise it will affect the accuracy of the summary. The legal names involved are given in Chinese)
                '''},
            {"role": "user", "content": text}
        ]
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
    
    
def extract_json_from_response(response):
    """
    从LLM的返回值中提取json
    """
    if response.startswith("json"):
        response = response[len("json"):].strip()
    if response.startswith("python"):
        response = response[len("python"):].strip()
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.startswith("```python"):
        response = response[len("```python"):].strip()
    if response.endswith("```"):
        response = response[:-len("```")].strip()

    # Parse the JSON string
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {e}")   
    
    

def process_text(text, llm_model):
    """
    单条文本的处理
    """
    response_content = get_response(text, llm_model=llm_model)
    response = extract_json_from_response(response_content)
    return response


def process_single_text(idx, text, llm_model):
    """
    处理单条文本的线程
    """
    print(f"Processing {idx+1}...")
    return process_text(text, llm_model=llm_model)   


def process_texts_from_file(input_file, output_file, value_file, llm_model):
    """
    使用多线程处理文本
    """
    with open(input_file, "r", encoding="utf-8") as file:
        datas = json.load(file)
    
    texts = [f"quesion:{data['title']}:{data['content']},type:{data['type']},answer:{data['answer']}" for data in datas]
    # 断点恢复
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            processed_data = json.load(file)
            processed_indices = {idx for idx, _ in enumerate(processed_data)}

    responses = processed_data if processed_indices else [None] * len(texts)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_text, idx, text, llm_model): idx for idx, text in enumerate(texts) if idx not in processed_indices}

        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                response = future.result()
                responses[idx] = response  # 保持原始数据的顺序
            except Exception as e:
                with open(output_file, "w", encoding="utf-8") as file:
                    json.dump(responses, file, ensure_ascii=False, indent=4)
                print(f"处理第{idx+1}条数据时出现错误: {e}")
    # 保存筛选结果提取values           
    save_to_file(datas,responses,output_file)
    analyze_json_values(output_file, value_file)
      
        
    
def save_to_file(datas,responses,output_file):
    """
    将结果保存在文件中
    """
    new_data = []
    index= 0
    for (data, response) in zip(datas, responses):
        if response and response.get("flag") is True:
                if data['metadata'] is None:
                    data['metadata'] = {}
                elif data['metadata'] == {}:
                    data['metadata'] = {'laws': ''}
                # 替换 metadata 中的 laws 字段
                if 'laws' not in data['metadata']:
                    data['metadata']['laws'] = ''
                data['metadata']['laws'] = response.get('laws')
                data['text_id'] = index
                index += 1
                new_data.append(data)
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)
    print(f"处理完成！筛选后的数据已保存到 {output_file}")
    
    
def analyze_json_values(input_file, output_file):
    """
    获取每个字段可能的值并保存在json中
    """
    field_values = defaultdict(lambda: defaultdict(list))
    with open(input_file, "r", encoding="utf-8") as file:
        datas = json.load(file)
    metadatas = [data.get('metadata') for data in datas if 'metadata' in data]
    
    for index, item in enumerate(metadatas):
        if isinstance(item, dict):  
            for key, value in item.items():
                if key == "text_id":
                    continue
                if isinstance(value, list):
                    # 将列表中的每个元素及其对应的 text_id 添加到字典中
                    for val in value:
                        field_values[key][val].append(index)
                else:
                    field_values[key][value].append(index)

    formatted_field_values = {
        key: [{"value": val, "text_id": ids} for val, ids in values.items()]
        for key, values in field_values.items()
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(formatted_field_values, file, ensure_ascii=False, indent=4)
    print(f"提取完成！values数据已保存到 {output_file}")
    

client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key  = "sk-c02056e042844ee1815cb64d8ede554f")

if __name__ == "__main__":
    # 参数设置
    llm_model = "qwen-plus-latest"
    data_dir = os.path.join(os.path.dirname(__file__), 'corpus')
    
    input_file = "qa.json"
    output_file = "qa_filtered.json"
    value_file = "qa_values.json"
    
    input_path = os.path.join(data_dir, input_file)
    output_path = os.path.join(data_dir, output_file)
    value_path = os.path.join(data_dir, value_file)
    # 批量处理
    process_texts_from_file(input_path, output_path, value_path, llm_model=llm_model)
    
    
    
    