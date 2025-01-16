from typing import Dict, List
from openai import OpenAI
from rag.config.config import Config



class Generator:
    def __init__(self, config: Config):
        self.config = config

    def _build_messages(self, query: str, retrieval_results: List[Dict], historys: List[Dict]) -> List[dict]:
        # 系统提示信息
        messages = [
            {
                "role": "system",
                "content": self.config["system_prompt"],
            }
        ]

        # 添加 Few-Shot 示例
        few_shot_examples = self.config["few_shot_examples"]
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example["query"]})
            messages.append({"role": "assistant", "content": example["answer"]})

        # 添加历史对话
        if historys:
            for history in historys:
                messages.append({"role": "user", "content": history["query"]})
                messages.append({"role": "assistant", "content": history["answer"]})

        # 构造当前 query 和检索结果
        context_list = ""
        for i, doc in enumerate(retrieval_results):
            context = f"\n文档集 {i+1}:\n- {doc}\n"
            print(context)
            context_list += context

        # Chain-of-Thought 提示
        chain_of_thought_prompt = self.config["chain_of_thought_prompt"]
        content_string = chain_of_thought_prompt.format(query=query, context_list=context_list)
        
        messages.append(
            {
                "role": "user",
                "content": content_string,
            }
        )
        return messages

    def generate(self, query: str, retrieval_results: List[Dict], historys: List[Dict]) -> str:
        """
        调用千问模型 API 生成答案。
        """
        if not query or not retrieval_results:
            return "No query or retrieval results provided."

        # 构造 messages 数据结构
        messages = self._build_messages(query, retrieval_results, historys)

        # 使用成功的调用方式构造请求体
        client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
        )
        generate_config = self.config["generate_config"]

        # 调用千问模型的 chat API
        completion = client.chat.completions.create(
            model=self.config["llm_model"],
            messages=messages,
            max_tokens=generate_config['max_tokens'],  
            temperature=generate_config["temperature"],  
            top_p=generate_config["top_p"],  
            n=generate_config["n"],  
        )

        # 提取生成的答案内容
        generated_text = completion.choices[0].message.content.strip()
        return generated_text
