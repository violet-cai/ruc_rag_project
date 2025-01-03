# -*- coding: utf-8 -*-
from openai import OpenAI
import json
import os
import sys

# 将项目根目录添加到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.database.milvus import MilvusClientWrapper
from rag.config.config import Config
from rag.database.chunk import chunk_data
import time

from pymilvus import (model,
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from tqdm import tqdm
# 导入union类型
from typing import Union, List


llm_model = 'gpt-4o-mini'
client = OpenAI(
    api_key='your_api_key',
)

config = Config()
# sparse_embedding_model = model.hybrid.BGEM3EmbeddingFunction(device='cpu', return_sparse=True, return_dense=False, return_colbert_vecs=False)
dense_embedding_model = model.DefaultEmbeddingFunction()
wrapper = MilvusClientWrapper(config)
batch_size = 5 # 批量处理数据的大小
topk = 15 # 检索时返回的文档数量
collection_name = "regulation_evaluate"

# llm筛选问题-答案对
def filter_single_qa(text)->bool:
    """
    llm筛选问答对,主要目的是去除qa数据中回答不好的问答对
    """
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content":
            '''你是一个精准判断问答对是否合格的机器人，你需要根据输入的内容判断该问答对是否合格,问题存在于"title"和
            "question"字段中,回答存在于"answer"字段中.问答对合格的标准如下：
            1. 回答内容不为空,且不是无意义的内容
            2. 问题和回答之间有明显的逻辑关系
            3. 问题和回答之间的内容不是无意义的重复
            4. 回答内容直接对问题进行了回答,而非让用户去查找答案
            如果问答对合格，请回答"True"，否则回答"False"
            示例如下：
            示例1：
            输入：
            {
                "qa_id": 1,
                "title": "公司准备开展来料加工业务",
                "question": "公司准备开展来料加工，经营范围内有货物进出口权、但是没有加工权，如果想要开展来料加工，需要办理哪些资料、还有需要取得哪些资质",
                "answer": "根据《中华人民共和国海关加工贸易货物监管办法》（海关总署令第219号公布，海关总署令第235号、240号、243号、247号、262号修正）及有关规定，企业开展加工贸易业务时应由经营企业向加工企业所在地主管海关办理加工贸易货物的手册设立手续，办理手册设立手续时请提供以下资料：1.《加工贸易加工企业经营状况及生产能力信息表》。2.经营企业委托加工的，应当提交经营企业与加工企业签订的委托加工合同。3.料件（成品）为贸易管制商品的，应提供相关部门的批文或许可证件：a.开展经信息产业部定点的全向型和外向型卫星电视接收设施生产企业开展卫星电视接收设施加工贸易业务的，需向主管海关提供国家颁发的卫星电视接收设施生产许可证件。b.开展关税配额农产品加工贸易手册业务的，企业应提供省级商务主管部门出具的业务批准文件c.开展列入《限制进口类可用作原料的固体废物目录》或《自动许可进口类可用作原料的固体废物目录》固体废物的加工贸易业务的，需向海关提供各种废物进口许可证（限制进口类可用作原料的固体废物目录）。4.《代理报关委托书/委托报关协议》（委托代理报关的）。5.首期加工贸易电子化手册设立的，还须提供以下资料：《加工贸易企业基本情况登记表》及其附页1份（附管理人员照片、身份证复印件、企业大门、主要生产场所、厂区照片）、外经部门批文、厂房租赁协议（或购买合同）、企业基本情况登记表等。6.开展异地加工的，企业需上传提供厂房自有证明或租赁合同（租赁厂房需征收风险担保金）7、如企业须缴纳风险担保的，需上传提供风险担保金计算清单。8、经营企业对外签订的合同。9、企业认为需要向海关提交的说明材料。",
            }
            输出为：
            True
            
            示例2：
            输入为：
            {
                "qa_id": 2,
                "title": "2024年1月-2024年12月可以从缅甸进口入关的农副产品有哪些",
                "question": "从缅甸进口入关的农副产品清单，哪些可以走大宗贸易",
                "answer": "您好！可登陆海关总署动植检司网站（http://dzs.customs.gov.cn）的“企业信息”栏目，查询获得我国检验检疫准入的新鲜水果、粮食、植物源性饲料种类及输出国家地区名录。获得我国检疫准入的缅甸农产品均可按规定正常申报进口。"
            }
            输出为：
            False
            
            一定注意只输出"True"或"False"，不能添加额外字段,输出后请检查是否符合要求，符合要求后再提交
            '''},
            {"role": "user", "content": text}
        ],
    )

    return response.choices[0].message.content == "True"

# 筛选问答对
def filter_qa(texts:Union[str, List[str]])->List[str]:
    """
    筛选问答对
    """
    if isinstance(texts, str):
        texts = [texts]
    return [text for text in texts if filter_single_qa(text)]

# 利用llm改写问题为查询：
def get_single_query(text:str):
    """
    利用llm改写问题为查询
    """

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content":
            '''你是一个精通问题重写的机器人，你需要根据输入的内容改写为查询，
            改写的查询需要简洁明了，包含输入内容的关键信息：
            示例如下：
            输入：
            示例1：
            {
                "qa_id": 1,
                "title": "公司准备开展来料加工业务",
                "question": "公司准备开展来料加工，经营范围内有货物进出口权、但是没有加工权，如果想要开展来料加工，需要办理哪些资料、还有需要取得哪些资质",
                "answer": "根据《中华人民共和国海关加工贸易货物监管办法》（海关总署令第219号公布，海关总署令第235号、240号、243号、247号、262号修正）及有关规定，企业开展加工贸易业务时应由经营企业向加工企业所在地主管海关办理加工贸易货物的手册设立手续，办理手册设立手续时请提供以下资料：1.《加工贸易加工企业经营状况及生产能力信息表》。2.经营企业委托加工的，应当提交经营企业与加工企业签订的委托加工合同。3.料件（成品）为贸易管制商品的，应提供相关部门的批文或许可证件：a.开展经信息产业部定点的全向型和外向型卫星电视接收设施生产企业开展卫星电视接收设施加工贸易业务的，需向主管海关提供国家颁发的卫星电视接收设施生产许可证件。b.开展关税配额农产品加工贸易手册业务的，企业应提供省级商务主管部门出具的业务批准文件c.开展列入《限制进口类可用作原料的固体废物目录》或《自动许可进口类可用作原料的固体废物目录》固体废物的加工贸易业务的，需向海关提供各种废物进口许可证（限制进口类可用作原料的固体废物目录）。4.《代理报关委托书/委托报关协议》（委托代理报关的）。5.首期加工贸易电子化手册设立的，还须提供以下资料：《加工贸易企业基本情况登记表》及其附页1份（附管理人员照片、身份证复印件、企业大门、主要生产场所、厂区照片）、外经部门批文、厂房租赁协议（或购买合同）、企业基本情况登记表等。6.开展异地加工的，企业需上传提供厂房自有证明或租赁合同（租赁厂房需征收风险担保金）7、如企业须缴纳风险担保的，需上传提供风险担保金计算清单。8、经营企业对外签订的合同。9、企业认为需要向海关提交的说明材料。",
            }
            输出为：
            {
                "qa_id": 1,
                "title": "公司准备开展来料加工业务",
                "question": "公司准备开展来料加工，经营范围内有货物进出口权、但是没有加工权，如果想要开展来料加工，需要办理哪些资料、还有需要取得哪些资质",
                "answer": "根据《中华人民共和国海关加工贸易货物监管办法》（海关总署令第219号公布，海关总署令第235号、240号、243号、247号、262号修正）及有关规定，企业开展加工贸易业务时应由经营企业向加工企业所在地主管海关办理加工贸易货物的手册设立手续，办理手册设立手续时请提供以下资料：1.《加工贸易加工企业经营状况及生产能力信息表》。2.经营企业委托加工的，应当提交经营企业与加工企业签订的委托加工合同。3.料件（成品）为贸易管制商品的，应提供相关部门的批文或许可证件：a.开展经信息产业部定点的全向型和外向型卫星电视接收设施生产企业开展卫星电视接收设施加工贸易业务的，需向主管海关提供国家颁发的卫星电视接收设施生产许可证件。b.开展关税配额农产品加工贸易手册业务的，企业应提供省级商务主管部门出具的业务批准文件c.开展列入《限制进口类可用作原料的固体废物目录》或《自动许可进口类可用作原料的固体废物目录》固体废物的加工贸易业务的，需向海关提供各种废物进口许可证（限制进口类可用作原料的固体废物目录）。4.《代理报关委托书/委托报关协议》（委托代理报关的）。5.首期加工贸易电子化手册设立的，还须提供以下资料：《加工贸易企业基本情况登记表》及其附页1份（附管理人员照片、身份证复印件、企业大门、主要生产场所、厂区照片）、外经部门批文、厂房租赁协议（或购买合同）、企业基本情况登记表等。6.开展异地加工的，企业需上传提供厂房自有证明或租赁合同（租赁厂房需征收风险担保金）7、如企业须缴纳风险担保的，需上传提供风险担保金计算清单。8、经营企业对外签订的合同。9、企业认为需要向海关提交的说明材料。",
                "query": "企业开展加工贸易业务需要提供的资料和资质"
            }
            示例2：
            输入为：
            {
                "qa_id": 2,
                "title": "中国对缅甸的进口农产品货物清单有哪些（哪些走大宗贸易）",
                "question": "缅甸这边能够出口到中国的农副产品有哪些，哪些可以走大宗贸易",
                "answer": "您好！可登陆海关总署动植检司网站（http://dzs.customs.gov.cn）的“企业信息”栏目，查询获得我国检验检疫准入的新鲜水果、粮食、植物源性饲料种类及输出国家地区名录。获得我国检疫准入的缅甸农产品均可按规定正常申报进口。",
            }
            输出为：
            {
                "qa_id": 2,
                "title": "中国对缅甸的进口农产品货物清单有哪些（哪些走大宗贸易）",
                "question": "缅甸这边能够出口到中国的农副产品有哪些，哪些可以走大宗贸易",
                "answer": "您好！可登陆海关总署动植检司网站（http://dzs.customs.gov.cn）的“企业信息”栏目，查询获得我国检验检疫准入的新鲜水果、粮食、植物源性饲料种类及输出国家地区名录。获得我国检疫准入的缅甸农产品均可按规定正常申报进口。",
                "query": "中国对缅甸的进口农产品货物和大宗贸易清单"
            }
    
            一定注意输出的"text_id","title"，"question"，"answer"与输入不变，只需添加"query"字段，"query"字段为改写后的查询，不能添加额外字段
            输出后请检查是否符合要求，符合要求后再提交
            '''},
            {"role": "user", "content": text}
        ],
    )
    
    return response.choices[0].message.content
    
def get_query(texts:Union[str, List[str]]):
    """
    获取查询
    """
    if isinstance(texts, str):
        texts = [texts]
    return [get_single_query(text) for text in texts]

# 检索结果
def get_single_retrieve(collection_name:str, query:str, k:int) -> List[dict]:
    """
    检索结果,bm25 + dense (onnx)
    """
    full_text_search_params = {"metric_type": "BM25"}
    dense = dense_embedding_model.encode_queries([query])
    dense_vec = dense[0]
    output_fields = ["text_id","content"]
    full_text_search_req = AnnSearchRequest(
        [query], "bm25_sparse_vector", full_text_search_params, limit=k
    )

    dense_search_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(
        [dense_vec], "dense_vector", dense_search_params, limit=k
    )
    
    result = wrapper.client.hybrid_search(
    collection_name,
    [full_text_search_req, dense_req],
    ranker=RRFRanker(),
    limit=k,
    output_fields=output_fields,
    )
    
    return result[0]

def get_retrieve(collection_name:str, queries:Union[str, List[str]], k:int) ->List[dict]:
    """
    检索结果
    [{
        query: str,
        reference_docs: [
            {
                text_id: str,
                content: str,
            }
        ]
    },
    {
        query: str,
        reference_docs: List[dict]
    }]
    """
    if isinstance(queries, str):
        queries = [queries]
    
    answers_list = []
    for query in queries:
        docs = get_single_retrieve(collection_name, query, k)
        answers = []
        for doc in docs:
            answer = {
            "text_id": doc["entity"]["text_id"],
            # "chunk_id": doc["entity"]["chunk_id"], # chunk_id字段添加
            "content": doc["entity"]["content"],
            # "score": doc["distance"],
            }
            answers.append(answer)
        
        answers_list.append({"query": query, 
                             "reference_docs": answers})
    return answers_list
        
# 利用大模型确定golden_docs
def evaluate_single_retrieve(text) -> List[dict]:
    """
    大模型为每个query的检索结果打分
    完全相关：3
    部分相关：2
    不相关：1
    """
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content":
            '''你是一个精准判断查询和检索内容是否存在相关性的机器人，你需要针对输入的查询和检索内容进行判断，
            并为每个检索文档打上相关性标签，相关性标签是评估检索结果的重要依据，标明了文档与查询的相关性。
            相关性标签分为如下三类：
            完全相关：文档与查询完全匹配，提供了较完整的答案。relevance_label = 3
            部分相关：文档与查询有部分相关性，但并不能完全回答查询。relevance_label = 2
            不相关：文档与查询完全不相关。relevance_label = 1
            示例如下：
            输入：
            示例：
            {
                "qa_id": 1
                "query": "广东省企业开展加工贸易业务需要提供的资料和资质",
                "reference_docs": [
                    {   
                        "text_id": 1,
                        "content":
                        "
                        现公告广东省内加工贸易企业合同备案（变更）、异地加工、内销等有关问题：一、加工贸易合同备案（变更）：1. 广东省内经营企业申请加工贸易合同备案（变更），不再提交《加工贸易业务批准证》等文件。2. 申请加工贸易合同备案需提交《加工贸易加工企业生产能力证明》。3. 加工贸易合同所对应的料件、成品及单耗等有关情况，由企业直接向海关申报。4. 涉及贸易管制商品的，需提交相关批准文件。5. 手册有效期原则上不超过一年，经批准可延长至两年，特殊行业按实际合同有效期备案。
                        二、异地加工贸易管理：广东省内企业间开展异地加工贸易业务，不再提交《加工贸易业务批准证》。
                        三、加工贸易内销：广东省内经营企业申请加工贸易货物内销，不再提交《加工贸易保税进口料件内销批准证》。
                        四、其他相关事项：广东省内企业录入合同备案（变更）、内销申请表等电子数据时，“批准文号”栏目统一填写“1111”。海关总署2013年9月2日。
                        "                
                    },
                    {
                        "text_id": 2,
                        "content":
                        "
                        为进一步规范限制类商品加工贸易管理，针对《商务部、海关总署2007年第44号公告》的执行问题，公告如下：
                        一是台账保证金征收环节，企业应在办理手册备案（变更）时缴纳。
                        二是台账保证金计征方法，A类（包括AA类）、B类企业按特定公式计算，深加工结转的限制类商品不计入计算范围。
                        三是深加工结转限制类商品管理，自2007年8月23日起，需单项备案并在商品名称首字节注明“[深]”。
                        四是电子帐册管理，电子帐册企业需备案台账手册，新增限制类商品需一次性备案。
                        五是电子手册管理，2007年8月23日前审批的电子手册按原规定执行，此后审批的需一次性备案料件和制成品。
                        六是信息化系统调整期的业务执行，H2000系统调整前，企业需按手工方式简易计征保证金台账。七是H2000系统修改，增加限制类标志，不同标志按不同规定计征台账保证金。
                        八是已征收的台账保证金待核销结案后退还。
                        "
                    },
                    {
                        "text_id": 3,
                        "content":
                        "
                        为落实党中央、国务院扩大高水平开放、深化“放管服”改革的决策部署，海关总署决定精简和规范部分加工贸易业务办理手续。
                        一是手册设立（变更）一次申报，取消备案资料库申报；
                        二是账册设立（变更）一次申报，取消商品归并关系申报；
                        三是外发加工一次申报，取消外发加工收发货记录；
                        四是深加工结转一次申报，取消事前申请和收发货记录；
                        五是余料结转一次申报，不再征收风险担保金；
                        六是内销征税一次申报，统一内销征税申报时限；
                        七是优化不作价设备监管，简化解除监管流程；
                        八是创新低值辅料监管，纳入保税料件统一管理。本公告自2020年1月1日起实施。
                        "
                    }
                ]
            }
            输出为：
            {
                "qa_id": 1
                "query": "广东省企业开展加工贸易业务需要提供的资料和资质",
                "reference_docs": [
                    {
                        "text_id": 1,
                        "content":
                        "
                        现公告广东省内加工贸易企业合同备案（变更）、异地加工、内销等有关问题：一、加工贸易合同备案（变更）：1. 广东省内经营企业申请加工贸易合同备案（变更），不再提交《加工贸易业务批准证》等文件。2. 申请加工贸易合同备案需提交《加工贸易加工企业生产能力证明》。3. 加工贸易合同所对应的料件、成品及单耗等有关情况，由企业直接向海关申报。4. 涉及贸易管制商品的，需提交相关批准文件。5. 手册有效期原则上不超过一年，经批准可延长至两年，特殊行业按实际合同有效期备案。
                        二、异地加工贸易管理：广东省内企业间开展异地加工贸易业务，不再提交《加工贸易业务批准证》。
                        三、加工贸易内销：广东省内经营企业申请加工贸易货物内销，不再提交《加工贸易保税进口料件内销批准证》。
                        四、其他相关事项：广东省内企业录入合同备案（变更）、内销申请表等电子数据时，“批准文号”栏目统一填写“1111”。海关总署2013年9月2日。
                        "
                        "relevance_label": 2                    
                    },
                    {
                        "text_id": 2,
                        "content":
                        "
                        为进一步规范限制类商品加工贸易管理，针对《商务部、海关总署2007年第44号公告》的执行问题，公告如下：
                        一是台账保证金征收环节，企业应在办理手册备案（变更）时缴纳。
                        二是台账保证金计征方法，A类（包括AA类）、B类企业按特定公式计算，深加工结转的限制类商品不计入计算范围。
                        三是深加工结转限制类商品管理，自2007年8月23日起，需单项备案并在商品名称首字节注明“[深]”。
                        四是电子帐册管理，电子帐册企业需备案台账手册，新增限制类商品需一次性备案。
                        五是电子手册管理，2007年8月23日前审批的电子手册按原规定执行，此后审批的需一次性备案料件和制成品。
                        六是信息化系统调整期的业务执行，H2000系统调整前，企业需按手工方式简易计征保证金台账。七是H2000系统修改，增加限制类标志，不同标志按不同规定计征台账保证金。
                        八是已征收的台账保证金待核销结案后退还。
                        "
                        "relevance_label": 1
                    },
                    {
                        "text_id": 3,
                        "content":
                        "
                        为落实党中央、国务院扩大高水平开放、深化“放管服”改革的决策部署，海关总署决定精简和规范部分加工贸易业务办理手续。
                        一是手册设立（变更）一次申报，取消备案资料库申报；
                        二是账册设立（变更）一次申报，取消商品归并关系申报；
                        三是外发加工一次申报，取消外发加工收发货记录；
                        四是深加工结转一次申报，取消事前申请和收发货记录；
                        五是余料结转一次申报，不再征收风险担保金；
                        六是内销征税一次申报，统一内销征税申报时限；
                        七是优化不作价设备监管，简化解除监管流程；
                        八是创新低值辅料监管，纳入保税料件统一管理。本公告自2020年1月1日起实施。
                        "
                        "relevance_label": 3
                    }
                ]
            }
            
            一定注意输出的""query_id"","query","text_id"与输入不变，只需在answers列表中每个文档中添加"label"字段，"label"字段为文档与查询的相关性标签，不能添加额外字段
            输出后请检查是否符合要求，符合要求后再提交
            '''},
            {"role": "user", "content": text}
        ],
    )
    
    return response.choices[0].message.content
    
def evaluate_retrieve(answers_list:List[dict]) -> List[dict]:
    """
    对检索数据集进行llm打分
    """
    # 对检索数据集进行llm打分
    for answers in answers_list:
        # 每个answer进行batch打分
        for i in range(0, len(answers['reference_docs']), batch_size):
            batch_answers = answers['reference_docs'][i:i+batch_size]
            cur_batch = {
                "qa_id": answers['qa_id'],
                "query": answers['query'],
                "reference_docs": batch_answers
            }
            # 转为string
            cur_batch = json.dumps(cur_batch, ensure_ascii=False, indent=4)
            response = evaluate_single_retrieve(cur_batch)
            # 转为dict
            response = json.loads(response)
            answers['reference_docs'][i:i+batch_size] = response['reference_docs']
    return answers_list

if __name__ == '__main__':
    
    data_dir = os.path.join(os.path.dirname(__file__), 'corpus')
    
    qa_file = "qa_filtered.json"
    regulation_file = "dataset.json"
    
    qa_file = os.path.join(data_dir, qa_file)
    regulation_file = os.path.join(data_dir, regulation_file)
    output_file = os.path.join(data_dir, "evaluate_data.json")
    
    with open(qa_file, 'r', encoding='utf-8') as file:
        data_qa = json.loads(file.read())
        
    with open(regulation_file, 'r', encoding='utf-8') as file:
        data_regulation = json.loads(file.read())
    
    time1 = time.time()
    # 构建数据
    texts = []
    for item in data_qa[:100]:
        text = {
                "qa_id": item.get('text_id'),
                "title": item.get('title'),
                "question": item.get('content'),
                "answer": item.get('answer')
        }
        texts.append(json.dumps(text, ensure_ascii=False, indent=4))
        
    # 筛选问答对
    texts = filter_qa(texts)
    time2 = time.time()
    print("filter time cost:", time2-time1)
    
    # 改写查询,耗时太长,考虑多线程和异步
    queries = get_query(texts)
    # 将string转为dict
    qa_with_query = [json.loads(query) for query in queries]
    time3 = time.time()
    print("rewrite query time cost:", time3-time2)
    
    # 获取qa_id和query
    queries = []
    queries_id = []
    for qa in qa_with_query:
        queries.append(qa.get('query'))
        queries_id.append(qa.get('qa_id'))
        
    
    # 构建检索数据集
    answers_list = get_retrieve(collection_name, queries, topk)
    for i,query in enumerate(queries):
        answers_list[i]['qa_id'] = queries_id[i]

    # 对检索结果打分,发现大模型对query打分结果大多为1,并且耗时太长,后续修改再使用,考虑多线程和异步 
    # answers_list = evaluate_retrieve(answers_list)
    # time4 = time.time()
    # print("llm evaluate time cost:", time4-time3)

    # 对answers_list添加qa_with_query对应qa_id的"answer"内容
    for i, qa in enumerate(qa_with_query):
        for answers in answers_list:
            if qa.get('qa_id') == answers.get('qa_id'):
                answers['title'] = qa.get('title')
                answers['question'] = qa.get('question')
                answers['answer'] = qa.get('answer')
                break
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(json.dumps(answers_list, ensure_ascii=False, indent=4))
    print("store evaluatr_data time cost:", time.time()-time3)

    
    