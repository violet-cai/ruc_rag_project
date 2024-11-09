import ollama


def generate(query_list, retrieved_list):
    prompt = "从给定的资料中，回答多个问题，并且综合多个问题的答案。如果问题答案无法从资料中获得，结合资料输出. 如果找到答案, 直接输出答案。"
    n = len(query_list)
    m = len(retrieved_list[0])
    for i in range(n):
        prompt += f"问题{i}:{query_list[i]}\n"
        for j in range(m):
            prompt += f"问题{i}的资料：{retrieved_list[i][j]} "
        prompt += "\n"
    prompt += "合并以上问题和资料，得到一个最终的答案，能回答以上所有问题"
    stream = ollama.generate(
        stream=True,
        model="qwen2.5:3b",
        prompt=prompt
    )
    for chunk in stream:
        if not chunk['done']:
            print(chunk['response'], end='', flush=True)
        else:
            print('\n')
            print('Note:done.')