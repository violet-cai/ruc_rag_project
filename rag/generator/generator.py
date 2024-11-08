import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, config, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, query_list, retrieved_list):
        n = len(query_list)
        k = len(retrieved_list[0])
        prompt = ""
        for i in range(n):
            prompt += "帮我结合给定的资料，回答多个问题，并且综合多个问题的答案。如果问题答案无法从资料中获得，输出结合给定的内容，无法回答问题. 如果找到答案, 就输出找到的答案。"
            prompt += f"问题{i}: {query_list[i]}"
            for j in range(k):
                prompt += f"内容{j}：{retrieved_list[i][j]}"
        prompt += f"合并以上内容，回答这{n}个问题"
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. Help me to answer the questions."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response