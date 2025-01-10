from pathlib import Path
import json
import pandas as pd

class ManualEvaluator:
    def __init__(self):
        self.scores_file = Path("rag/evaluater_manual/scores.json")
        self.load_scores()
        
    def load_scores(self):
        if self.scores_file.exists():
            with open(self.scores_file, "r", encoding="utf-8") as f:
                self.scores = json.load(f)
        else:
            self.scores = {}
            
    def save_scores(self):
        with open(self.scores_file, "w", encoding="utf-8") as f:
            json.dump(self.scores, f, indent=2, ensure_ascii=False)
            
    def add_score(self, conversation_id, scores):
        self.scores[conversation_id] = scores
        self.save_scores()
        
    def get_statistics(self):
        if not self.scores:
            return None
            
        df = pd.DataFrame.from_dict(self.scores, orient='index')
        stats = {
            "平均分": df.mean(),
            "标准差": df.std(),
            "最高分": df.max(),
            "最低分": df.min()
        }
        return pd.DataFrame(stats)