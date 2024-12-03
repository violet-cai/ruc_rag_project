from typing import List

import torch
from rag.config.config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config["generate_model"])
        self.model = AutoModelForCausalLM.from_pretrained(
            config["generate_model"], torch_dtype="auto", device_map="auto"
        )

    def _build_prompt(self, query_list: List[str], reranked_list: List[List[str]]) -> str:
        prompt = (
            "You are an expert assistant. Please answer the following questions based on the provided references:\n\n"
        )

        for i, (query, refs) in enumerate(zip(query_list, reranked_list), 1):
            prompt += f"Question {i}: {query}\nReferences:\n"
            for ref in refs:
                prompt += f"- {ref}\n"
            prompt += "\nAnswer:\n"

        prompt += (
            "Based on the above questions and answers, please provide a comprehensive final summary:\nFinal Summary:\n"
        )
        return prompt

    def generator(self, query_list: List[str], reranked_list: List[List[str]]) -> str:
        if not query_list or not reranked_list:
            return "No questions or references provided."
        prompt = self._build_prompt(query_list, reranked_list)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,  # 根据需要调整最大长度
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_p=0.95,
                temperature=0.7,
                early_stopping=True,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_summary_start = "Final Summary:"
        final_summary_index = generated_text.find(final_summary_start)
        if final_summary_index != -1:
            final_summary = generated_text[final_summary_index + len(final_summary_start) :].strip()
        else:
            final_summary = "Could not generate a final summary."
        return final_summary
