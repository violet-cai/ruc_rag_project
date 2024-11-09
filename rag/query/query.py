import ollama


def parse_query(query):
    prompt = """
        你的任务是参考已有信息来解决用户问题。
        通过将用户问题拆分成能够通过搜索回答的多个子问题，每个搜索的问题应该是一个单一问题。
        再根据每个子问题衍生1-3个提问，用于更好的补充子问题。
        你的任务是参考已有信息来解决用户问题。
        通过将用户问题拆分成能够通过搜索回答的多个子问题，每个搜索的问题应该是一个单一问题。
        再根据每个子问题衍生1-3个提问，用于更好的补充子问题。
        
        输出为以下的JSON格式：
        {
            {
                "query1":<子问题描述>,
                "reference1":<衍生提问1>,
                "reference2":<衍生提问2>,
                "reference3":<衍生提问3>
            },
            {    
                "query2":<子问题描述>,
                "reference1":<衍生提问1>,
                "reference2":...
            },
            {
                "query3":...
                ...
            }
        }
        """
    stream = ollama.chat(
        model='qwen2.5:3b',
        messages=[
            {'role': 'user', 'content': query},
            {'role': 'system', 'content': prompt}
        ],
        stream=True,
    )
    res = ''
    for chunk in stream:
        res += chunk['message']['content']
    return res
