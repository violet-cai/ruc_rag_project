import ollama


def generate(query_list, retrieved_list):
    prompt = "需要你从给定的资料中，回答一个或多个问题，并且综合这些问题的答案。根据以下资料生成答案"
    n = len(query_list)
    m = len(retrieved_list[0])
    for i in range(n):
        prompt += f"问题{i}:{query_list[i]}\n"
        for j in range(m):
            prompt += f"问题{i}的资料：{retrieved_list[i][j]} "
        prompt += "\n"
    prompt += "如果资料与问题有关，根据以上问题和资料，得到一个能回答以上这些问题的最终答案"
    stream = ollama.generate(
        stream=True,
        model="qwen2.5:3b",
        prompt=prompt
    )
    for chunk in stream:
        if not chunk['done']:
            print(chunk['response'], end='', flush=True)
        else:
            pass
