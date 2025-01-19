import requests
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from rag.config.config import Config

config = Config()
# 定义服务端地址
SERVER_URL = config["server_url"]

# 调用接口
def _answer_api_show(query):
    url = f"{SERVER_URL}/get_answer_show"
    response = requests.post(
        url,
        json={"query": query}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
def _answer_api(query):
    url = f"{SERVER_URL}/get_answer"
    response = requests.post(
        url,
        json={"query": query}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

    
    
def get_answer(query):
    """
    调用示例
    """
    response = _answer_api_show(query)
    if response is None:
        print("回答问题出错，请检查服务端状态或查询内容。")
        return None
    new_query = response.get("query", "N/A")
    print("您当前的问题是：", new_query)
    
    docs = response.get("docs", "N/A")
    for i, doc in enumerate(docs):
        context = f"\n文档集 {i+1}:\n- {doc}\n"
        print(context)
    
    answer = response.get("answer", "N/A")
    print("生成的答案：", answer)
    return answer

def interactive_mode():
    """
    与用户在终端中交互演示
    """
    print("欢迎使用海关智能坐席助手！输入您的问题，或输入 'exit' 或 'quit' 退出。")
    while True:
        # 获取用户输入
        query = input("\n请输入您的问题：").strip()
        
        # 退出条件
        if query.lower() in ["exit", "quit"]:
            print("感谢使用，再见！")
            break

        # 调用服务端接口
        if query:
            print("\n正在处理您的问题，请稍候...")
            result = get_answer(query)
            if result:
                print("\n处理完成！")
            else:
                print("\n处理失败，请重试。")
        else:
            print("输入不能为空，请重新输入。")
            
# 启动客户端
if __name__ == "__main__":
    # 定义查询
    # query = "如果在准备材料的过程中遇到问题，应该联系谁呢？"
    interactive_mode()
