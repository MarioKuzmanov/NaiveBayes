import pandas as pd
import numpy as np


class Processor(object):
    def __init__(self):
        label2sentiment = {"__label__1": "negative", "__label__2": "positive"}
        labels, texts = [], []
        length = len("__label__1")
        with open("large_data/train.ft.txt", "rt", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                labels.append(label2sentiment[line[: length].strip()])
                texts.append(line[length:].strip())
        assert len(labels) == len(texts)

        labels, texts = np.array(labels), np.array(texts)

        labels = np.random.choice(labels, 30000)
        texts = np.random.choice(texts, 30000)

        self.df = pd.DataFrame({"sentiment": labels, "text": texts})

        print("Dataset Build!")
