import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import spacy
import pandas as pd
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import json
from preprocessor_large_data import Processor


class NB:
    def __init__(self, smoothing=1.0, small=True):
        self.smoothing = smoothing
        self.is_small = small
        self.nlp = spacy.load("en_core_web_sm")

        if small:
            df1 = pd.read_csv("small_data/sentimentdataset.csv", sep=",")
            df1["Sentiment"] = df1["Sentiment"].str.lower()
            df1 = df1.rename(columns={"Text": "text", "Sentiment": "sentiment"})
            df1 = df1[df1["sentiment"].isin(("positive", "negative", "neutral"))]

            df2 = pd.read_csv("small_data/tweets.csv", sep=",")
            df2 = df2.rename(columns={"airline_sentiment": "sentiment"})

            self.df = pd.concat([df1, df2])
        else:
            processor = Processor()
            self.df = processor.df

        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.label2total, self.naive_bayes, self.vocab, self.prior_prob = None, None, None, None

    def train(self):
        X, Y = np.array(self.df["text"]), np.array(self.df["sentiment"])

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, train_size=.8, test_size=.2)

        self.prior_prob = {label: 0.0 for label in sorted(set(self.Y_train))}

        # setup prior probabilities
        for label in self.prior_prob: self.prior_prob[label] = len(self.Y_train[self.Y_train == label]) / len(
            self.Y_train)

        processed_lines, vocab = defaultdict(list), set()

        # {"positive":count, "negative": count , "neutral": count}
        self.label2total = defaultdict(int)
        self.vocab = set()

        for id, text in enumerate(self.X_train):
            line = self.nlp(str(text))
            condition = lambda \
                    x: not x.is_punct and not x.is_digit and not x.is_space and not x.like_num and not x.like_url and not x.like_email

            # binary NB (optimization for Sentiment Analysis)
            line_tokens = {token.lower_.strip(string.punctuation) for token in line if condition(token)}
            processed_lines[self.Y_train[id]].append(line_tokens)
            self.vocab = self.vocab.union(line_tokens)

        self.vocab = sorted(self.vocab)

        # { word1 :  { "positive" : count , "negative": count, "neutral" : count } }
        # smoothing = 1.0
        self.naive_bayes = {w: {label: self.smoothing for label in list(self.prior_prob.keys())} for w in self.vocab}
        for label, lines in processed_lines.items():
            for line in lines:
                for w in line:
                    self.naive_bayes[w][label] += 1
                    self.label2total[label] += 1
        for label in self.label2total:
            self.label2total[label] += (len(self.vocab) * self.smoothing)

    def save(self):
        Y_hat = [self.predict(x_test)[0] for x_test in self.X_test]
        f1 = f1_score(self.Y_test, Y_hat, average="macro")

        model_settings = {"F1": f1, "NaiveBayes": self.naive_bayes, "TotalCounts": self.label2total,
                          "Priors": self.prior_prob,
                          "Vocab": self.vocab}

        filename = "models/nb_small.json" if self.is_small else "models/nb_large.json"

        with open(filename, "w", encoding="utf8") as f:
            json.dump(model_settings, f)

    def predict(self, text):
        line = self.nlp(text)
        condition = lambda \
                x: not x.is_punct and not x.is_digit and not x.is_space and not x.like_num and not x.like_url and not x.like_email
        test_label2prob = {label: 1 for label in list(self.label2total.keys())}
        for w in line:
            if condition(w):
                w = w.lower_.strip(string.punctuation)
                if w in self.vocab:
                    for label, count in self.naive_bayes[w].items():
                        addition = 0
                        if count > 0:
                            addition = np.log10(count / self.label2total[label])
                        test_label2prob[label] += addition
        # converting the log10 back
        for label in test_label2prob:
            test_label2prob[label] = 10 ** (np.log10(self.prior_prob[label]) + (test_label2prob[label]))
        y_hat = sorted(test_label2prob.items(), key=lambda x: x[1], reverse=True)[0][0]
        return y_hat, test_label2prob

    def load(self, nlp, label2total, vocab, naive_bayes, prior_prob):
        self.nlp = nlp
        self.label2total = label2total
        self.vocab = vocab
        self.naive_bayes = naive_bayes
        self.prior_prob = prior_prob


if __name__ == "__main__":
    # nb = NB(smoothing=1.0, small=True)
    nb = NB(smoothing=1.0, small=False)
    nb.train()
    nb.save()

    
    
    
    

