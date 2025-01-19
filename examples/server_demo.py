from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os,sys
import time
from pymilvus import connections

# 初始化 FastAPI 应用
app = FastAPI()
# 记录服务端启动时间
START_TIME = time.time()
class QueryRequest(BaseModel):
    query: str
    
# 初始化 RAG 组件
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from rag.config.config import Config
from rag.retriever.utils import get_retriever
from rag.reranker.utils import get_reranker
from rag.generator.utils import get_generator

config = Config()
retriever = get_retriever(config)
reranker = get_reranker(config)
generator = get_generator(config)

MAX_HISTORY_LENGTH = config["MAX_HISTORY_LENGTH"]  # 设置最大长度
# 全局变量存储 historys
historys = [
    {
        "query": "您好，我想了解一下关于中国澳门制造的保健食品在华注册备案的事宜。",
        "answer": "您好！根据相关规定，对于进口保健食品等18类食品的境外生产企业，采用“官方推荐注册”模式。也就是说，澳门特别行政区的主管当局需要对推荐注册的企业进行审核检查，并向海关总署推荐注册。",
    },
    {
        "query": "那是不是只有生产企业才能提交注册备案呢？如果是一般的贸易企业或者代理商可以吗？",
        "answer": "根据规定，必须是由所在国家或地区的主管当局推荐并提交申请材料，因此一般贸易企业或代理商本身没有直接提交注册备案的权利。不过，这些企业可以协助生产企业准备相关材料。",
    },
    {
        "query": "明白了，那具体需要提交哪些材料呢？",
        "answer": "主要包括：所在国家或地区主管当局的推荐函、企业名单与企业注册申请书、企业的身份证明文件（如营业执照）、主管当局的声明以及对相关企业的审查报告等。",
    },
]

def update_query(query):
    """
    更新query
    """
    global historys
    print("before query modified: ", query)
    query = retriever.update_query(query, historys)

    return query


def get_docs(query):
    """
    获取经过retrieve和rerank的文档
    """
    global historys
    retrieved_list = retriever.retrieve(query)
    retrieved_list_keywords = retriever.retrieve_with_keywords(query)
    retrieved_list_engine = retriever.retrieve_with_engine(query)
    # 与其他retrieved_list合并重排
    # retrieved_list = retrieved_list + retrieved_list_keywords
    # retrieved_list = retrieved_list + retrieved_list_engine
    reranked_list = reranker.rerank(query, retrieved_list)
    
    return reranked_list


def get_answer(query):
    """
    根据query获取答案
    """
    global historys 
    query = update_query(query)
    history = {}
    history["query"] = query
    print("after query modified: ", query)
    
    reranked_list = get_docs(query)
    answer = generator.generate(query, reranked_list, historys)
    
    history["answer"] = answer
    historys.append(history)  # 更新 historys
    
    # 限制 historys 长度
    if len(historys) > MAX_HISTORY_LENGTH:
        historys.pop(0)  # 删除最早的记录
    return query, reranked_list, answer


# 定义 API 接口
@app.post("/get_answer_show")
async def get_answer_api(request: QueryRequest):
    query = request.query
    try:
        new_query, docs, answer = get_answer(query)
        return {
            "query": new_query,
            "docs": docs,
            "answer": answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer")
async def get_answer_api(request: QueryRequest):
    query = request.query
    try:
        new_query, docs, answer = get_answer(query)
        return {
            "answer": answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """
    返回服务端的当前状态
    """
    return {
        "status": "running",
        "uptime": time.time() - START_TIME,  # 服务端运行时间（秒）
        "message": "服务端已启动并运行中"
    }

# 添加关闭服务端时的处理逻辑
@app.on_event("shutdown")
def shutdown_event():
    """
    服务端关闭时执行的操作
    """
    print("服务端正在关闭...")

    # 关闭 Milvus 连接
    print("关闭 Milvus 连接...")
    connections.disconnect("default")  # 关闭默认连接

    # 释放 retriever、reranker 和 generator 的资源
    if hasattr(retriever, "close"):
        print("释放 retriever 资源...")
        retriever.close()
    if hasattr(reranker, "close"):
        print("释放 reranker 资源...")
        reranker.close()
    if hasattr(generator, "close"):
        print("释放 generator 资源...")
        generator.close()



# 启动服务示例
if __name__ == "__main__":
    uvicorn.run(app, host=config["server_host"], port=config["server_port"])