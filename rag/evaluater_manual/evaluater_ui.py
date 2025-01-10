import streamlit as st
import json
import pandas as pd
from pathlib import Path
import yaml
from evaluater_manual import ManualEvaluator

class ManualEvaluationUI:
    def __init__(self):
        self.load_config()
        self.load_data()
        self.load_rating_standards()
        self.current_idx = 0
        self.scores = {}
        self.score_handler = ManualEvaluator()
        
    def load_config(self):
        config_path = Path("rag/config/basic_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
    
    def load_data(self):
        data_path = Path("rag/data/corpus/custom_conversations_20250109_result.json")
        with open(data_path, "r", encoding="utf-8") as f:
            self.conversations = json.load(f)
            
    def load_rating_standards(self):
        standards_path = Path("rag/data/corpus/rating_standard.md")
        with open(standards_path, "r", encoding="utf-8") as f:
            self.rating_standards = f.read()
            
    def render_conversation(self, idx):
        conv = self.conversations[idx]
        st.subheader(f"对话 {idx + 1}/{len(self.conversations)}")
        
        # 显示原始查询
        st.markdown("### 用户查询")
        st.write(conv["query"])
        
        # 显示检索结果
        st.markdown("### 检索结果")
        for i, result in enumerate(conv["retrieve_results"]):
            st.text(f"检索结果 {i+1}:")
            st.write(result)
            
        # 显示重排结果
        st.markdown("### 重排结果")
        for i, result in enumerate(conv["rerank_results"]):
            st.text(f"重排结果 {i+1}:")
            st.write(result)

        # 展示原始answer
        st.markdown("### 原始答案")
        st.write(conv["answer"])

        # 显示生成结果
        st.markdown("### 生成结果")
        st.write(conv["generation"])
            
    def render_scoring(self, idx):

        # 获取当前对话的评分
        if str(idx) in st.session_state.scores.keys():
            saved_scores = st.session_state.scores[str(idx)]
            st.markdown("### 该对话已评分，以下为已保存的评分，可进行手工调整:")
        else:
            saved_scores = {
                "检索相关性": 3,
                "检索准确性": 3,
                "生产相关性": 3,
                "生成忠实度": 3,
                "生成正确性": 3,
                "噪声鲁棒性": 3
            } 
            st.markdown("### 请对该对话进行评分:")
        
        # 从rating_standard.md中提取评分标准
        scores = {
            "检索相关性": st.slider("检索相关性得分", 1, 5, saved_scores["检索相关性"], key=f"ret_rel_{idx}"),
            "检索准确性": st.slider("检索准确性得分", 1, 5, saved_scores["检索准确性"], key=f"ret_acc_{idx}"),
            "生产相关性": st.slider("生产相关性得分", 1, 5, saved_scores["生产相关性"], key=f"gen_rel{idx}"),
            "生成忠实度": st.slider("生成忠实度得分", 1, 5, saved_scores["生成忠实度"], key=f"gen_fath{idx}"),
            "生成正确性": st.slider("生成正确性得分", 1, 5, saved_scores["生成正确性"], key=f"gen_right{idx}"),
            "噪声鲁棒性": st.slider("噪声鲁棒性得分", 1, 5, saved_scores["噪声鲁棒性"], key=f"noise_rub{idx}")
        }

        # 保存按钮
        if st.button("保存评分", key=f"save_{idx}"):
            st.session_state.scores[str(idx)] = scores
            self.score_handler.add_score(str(idx), scores)
            st.success("评分已保存!")
            
    def calculate_average_scores(self):
        if not st.session_state.scores:
            return None
            
        df = pd.DataFrame.from_dict(st.session_state.scores, orient='index')
        avg_scores = df.mean()
        return avg_scores

    def run(self):
        st.title("RAG系统手工评分界面")

        # 初始化会话状态变量
        if "current_idx" not in st.session_state:
            st.session_state.current_idx = 0
        if "scores" not in st.session_state:
            st.session_state.scores = self.score_handler.scores
        # st.write(st.session_state.scores)

        # 显示评分标准
        with st.expander("评分标准"):
            st.markdown(self.rating_standards)

        # 导航控制
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("上一条") and st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
        with col2:
            st.write(f"当前: {st.session_state.current_idx + 1}/{len(self.conversations)}")
        with col3:
            if st.button("下一条") and st.session_state.current_idx < len(self.conversations) - 1:
                st.session_state.current_idx += 1

        # 显示当前对话和评分界面
        self.render_conversation(st.session_state.current_idx)
        self.render_scoring(st.session_state.current_idx)

        # 显示总体评分统计
        st.markdown("### 评分统计")
        avg_scores = self.calculate_average_scores()
        if avg_scores is not None:
            st.dataframe(avg_scores)

if __name__ == "__main__":
    app = ManualEvaluationUI()
    app.run() 