import json
import random


class Item:
    def __init__(self, item_dict):
        self.id = item_dict.get("id", None)
        self.question = item_dict.get("question", None)
        self.golden_answers = item_dict.get("golden_answers", [])
        self.choices = item_dict.get("choices", [])
        self.metadata = item_dict.get("metadata", {})
        self.output = item_dict.get("output", {})

    def update_output(self, key, value):
        self.output[key] = value


class Dataset:
    def __init__(self, config, dataset_path, data=None):
        self.config = config
        self.dataset_path = dataset_path
        if data is None:
            self.data = self._load_data(dataset_path)
        else:
            self.data = data

    def _load_data(self, dataset_path):
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item_dict = json.loads(line)
                item = Item(item_dict)
                data.append(item)
        if self.sample_num is not None:
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[: self.sample_num]

        return data
