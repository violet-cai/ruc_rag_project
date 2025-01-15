from typing import List

import torch
from rag.config.config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config["generate_model"])
        self.model = AutoModelForCausalLM.from_pretrained(config["generate_model"], torch_dtype=torch.float16)
        self.prompt = config["generate_prompt"]
        self.generate_cofig = config['generate_config']

    def _build_prompt(self, query: str, reranked_list: List[str]) -> str:
        prompt = self.prompt
        prompt += f"问题: {query}\n参考文档:\n"
        for ref in reranked_list:
            prompt += f"- {ref}\n"
        prompt += "\n答案是："
        return prompt

    def generate(self, query: str, reranked_list: List[str]) -> str:
        if not query or not reranked_list:
            return "No question or references provided."
        prompt = self._build_prompt(query, reranked_list)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        generate_config = self.generate_cofig
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=generate_config['max_length'],
                num_return_sequences=generate_config['num_return_sequences'],
                no_repeat_ngram_size=generate_config['no_repeat_ngram_size'],
                top_p=generate_config['top_p'],
                temperature=generate_config['temperature'],
                num_beams=generate_config['num_beams'],
                early_stopping=generate_config['early_stopping'],
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_summary_start = "答案是："
        final_summary_index = generated_text.find(final_summary_start)
        final_summary = "\n\n"
        if final_summary_index != -1:
            final_summary = generated_text[final_summary_index + len(final_summary_start) :].strip()
        else:
            final_summary = "Could not generate a final summary."
        return final_summary
