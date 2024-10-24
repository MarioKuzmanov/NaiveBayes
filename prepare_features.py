import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import spacy
import numpy as np
import pandas as pd


class FeatureExtractor(object):

    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

        self.pos_lexicon, self.neg_lexicon = set(), set()

        with open(f"small_data/SentiWordNet_3.0.0.txt", "rt", encoding="utf8") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    tokens = line.split("\t")
                    assert len(tokens) == 6, "Not consistent format"
                    pos, neg, targets = tokens[2], tokens[3], tokens[4]
                    words = targets.split(" ")
                    for w in words:
                        w = w[: w.index("#")]
                        if float(pos) > float(neg):
                            self.pos_lexicon.add(w)
                        elif float(pos) < float(neg):
                            self.neg_lexicon.add(w)
        self.df_prepared = None

    def prepare_data(self):
        df = pd.read_csv("small_data/tweets.csv", sep=",")
        df = df[df["airline_sentiment"].isin(("positive", "negative"))]
        x = df[df["airline_sentiment"].isin(("positive", "negative"))]["text"].to_numpy()

        x1, x2, x3, x4, x5, x6 = [], [], [], [], [], []
        for example in x:
            ex1, ex2, ex3, ex4, ex5, ex6 = self._to_features(example)
            x1.append(ex1), x2.append(ex2), x3.append(ex3), x4.append(ex4), x5.append(ex5), x6.append(ex6)

        self.df_prepared = pd.DataFrame({"POS": x1, "NEG": x2, "HAS_NO": x3, "HAS_EXCLAMATION": x4, "WORDS": x5,
                                         "PRONOUNS": x6,
                                         "LABEL": df["airline_sentiment"]})

    def _to_features(self, example):
        line = self.nlp(example)

        # positive words, negative words, neutral words
        pos_count, neg_count = 0, 0
        # whether it contains no
        has_no = 0
        # whether it contains exclamation mark
        has_exclamation = 0
        # word count, normalized
        word_count = np.log(len(line))
        # pronouns
        count_pron = 0

        for token in line:
            if token.pos_ == "PRON": count_pron += 1

            if token.text.lower() == "no":
                has_no = 1
            elif token.text.lower() == "!":
                has_exclamation = 1

            if token.text in self.pos_lexicon:
                pos_count += 1
            elif token.text in self.neg_lexicon:
                neg_count += 1

        return pos_count, neg_count, has_no, has_exclamation, word_count, count_pron
