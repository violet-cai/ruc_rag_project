# 数据类
class EvaluationData:
    def __init__(self, pred, golden_answers, retrieval_result=None):
        self.pred = pred
        self.golden_answers = golden_answers
        self.retrieval_result=retrieval_result
        self.evaluation_scores = {}

    def update_evaluation_score(self, metric, score):
        self.evaluation_scores[metric] = score

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=4)

    # 可迭代支持：返回一个包含 pred 和 golden_answers 的元组
    def __iter__(self):
        return iter((self.pred, self.golden_answers))