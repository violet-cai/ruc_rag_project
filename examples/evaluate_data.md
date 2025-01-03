# 评估数据集

evaluate_data_beta_filtered.json。


## 评估数据集格式

**query**为基于**title**和**question**的查询,**reference_docs**为每个查询的参考文档集，
每个参考文档集包含**text_id**,**content**,**relevance_label**三个字段
**text_id**为文档id,**content**为文档内容,**relevance_label**为文档与查询的相关性标签

**reference_docs**包含10个文档，**score**字段为reference_docs中文档的相关性得分加和，并且evaluate_data_beta_filtered.json按照score从大到小排序，最低score为4

受限于各种资源，该版评估集一共37条，后续会继续更新增加

```python
# 如下格式
[
    {
        "query": "出口水果果园注册是否需要GAP认证",
        "reference_docs": [
            {
                "text_id": 847,
                "content": "海关总署公告2022年第41号（关于进口柬埔寨鲜食龙眼植物检疫要求的公告）:GACC将根据有害生物发生动态和截获",
                "relevance_label": 0
            },
            {
                "text_id": 1522,
                "content": "海关总署公告2018年第101号（关于进口吉尔吉斯斯坦鲜食甜瓜植物检疫要求的公告）:GACC将派员审核果园、包装厂管理情况，并定期回顾性审查检疫要求。",
                "relevance_label": 1
            },
           ...
        ],
        "qa_id": 173,
        "title": "出境新鲜果园注册申报",
        "question": "出境新鲜果园注册申报出口水果是否需要果园做GAP认证以后才可以办理。",
        "answer": "出境新鲜水果果园注册登记、出口检验检疫根据《出境水果检验检疫监督管理办法》等规定办理，对于GAP认证主要根据拟输往国家或地区官方要求确认。",
        "score": 13
    },
    {
        "query": "《进出口货物征免税申请表》和《减免税货物税款担保申请表》中的“进（出）口岸”如何填写",
        "reference_docs": [
            {
                "text_id": 38,
                "content": "中华人民共和国海关对平潭综合实验区监管办法（试行）（海关总署第208号令）:企业需及时报告海关监管货物遭遇灾害等情况，因不可抗力导致货物损坏、损毁、灭失的，海关予以核销或继续",
                "relevance_label": 0
            },
            {
                "text_id": 2055,
                "content": "海关总署2008年第86号公告（关于公布2009年1月1日起新增香港澳门享受零关税货物原产地标准表及相关事宜）:《香港原产货物标准表》和《澳门原产货物标准表》中使用了简化的货物名称，其范围与2008年《中华人民共和国进出口税则》中相应税号的货品一致。二、修改海关总署2006年第79号公告《享受货物贸易优惠措施的香港货物原产地标准表》中的“氨苄青霉素制剂”等27项货物和海关总署2007年第30号公告《2007年7月1日起新增香港享受零关税货物原产地标准表》中的“已配剂量含有磺胺类的药品”原产地标准（见附件3），新修改原产地标准自2009年1月1日起执行。特此公告。",
                "relevance_label": 0
            },
            ...
        ],
        "qa_id": 520,
        "title": "进出口货物征免税申请时“进（出）口岸怎么填写",
        "question": "我公司正在申请减免税业务，其中《进出口货物征免税申请表》和《减免税货物税款担保申请表》的“进（出）口岸”栏目的内容应如何填写？",
        "answer": "您好，针对您的问题，答复如下：《进出口货物征免税申请表》和《减免税货物税款担保申请表》中“进（出）口岸”栏目的填写内容，应根据货物实际进出境的口岸海关，填报海关规定的《关区代码表》中相应口岸海关的名称及代码。特殊情况填报要求如下：进口转关运输货物填报货物进境地海关名称及代码，出口转关运输货物填报货物出境地海关名称及代码。进出海关特殊监管区域或保税监管场所的货物，填报海关特殊监管区域或保税监管场所所在的海关名称及代码。结转的减免税货物，填写转出企业主管海关名称及代码。上述报表格式文本见海关总署公告2021年第16号（关于《中华人民共和国海关进出口货物减免税管理办法》实施有关事项的公告）附件。感谢您对海关工作的关注和支持！",
        "score": 8
    },
    ...
]
```

## 相关性打分标准

1. 完全相关：文档内容与查询较为匹配，文档内容在回答查询上具有强逻辑关系，能较完整回答查询或者提供了重要参考。relevance_label = 3

2. 部分相关：文档内容与查询有部分相关性，能提供一定程度参考，能回答部分查询的内容。relevance_label = 2

3. 微弱相关：文档内容与查询有微弱联系，能提供少量参考，能回答少部分查询内容。relevance_label = 1

4. 不相关：文档与查询完全不相关，无法提供任何与查询相关的信息。relevance_label = 0

## 如何评估

见分支examples/retrieve_evaluate_demo.py

更改
```python

path_evaluate_data = 'rag/data/corpus/evaluate_data_beta_filtered.json'

```

examples/retrieve_evaluate_demo.py 主要按照 examples/evaluate_data.py 格式进行评估

``

pred、retrieval_result格式如下

```python

pred:list 保存评估集的answer

retrieval_result: list[list[dict]] 保存每个query的实际检索结果

```

1. 若是评估检索

EvaluationData的golden answer为reference_docs

2. 若是评估生成

EvaluationData的golden answer为answer

pred、retrieval_result、golden_answer均一一对应于每个query
