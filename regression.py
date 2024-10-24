import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from prepare_features import *


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.a1 = nn.Linear(input_size, output_size)

    def forward(self, X):
        return self.a1(X)


class UseRegression(object):
    def __init__(self, df):
        label2idx = {"positive": 1.0, "negative": 0.0}

        X, Y = torch.tensor(df[df.columns.values[: -1]].to_numpy()).float(), torch.tensor(
            [label2idx[label] for label in df["LABEL"]]).long()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, train_size=.8)
        self.regressor = None
        self.f1 = None

    def fit(self, epochs=3000):

        self.regressor = LogisticRegression(input_size=self.X_train.shape[1], output_size=2)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.regressor.parameters(), lr=0.005)

        for i in range(epochs):
            optimizer.zero_grad()
            y_hat = self.regressor.forward(self.X_train)
            loss = loss_fn(y_hat, self.Y_train)
            loss.backward()
            optimizer.step()
            if i % 100 == 0: print(round(loss.item(), 3))

    def evaluate(self):
        assert self.regressor is not None, "You need to train model first."
        self.f1 = f1_score(torch.argmax(self.Y_test, dim=1).detach().numpy(),
                           np.where(self.regressor.forward(self.X_test).detach().numpy() >= 0.5, 1, 0),
                           average="macro")
        return self.f1

    def save(self):
        if self.f1 is None:
            self.f1 = f1_score(self.Y_test.detach().numpy(),
                               np.where(
                                   torch.argmax(self.regressor.forward(self.X_test), dim=1).detach().numpy() >= 0.5, 1,
                                   0),
                               average="macro")
        torch.save(
            {"F1": self.f1, "STATE": self.regressor.state_dict(), "InputSize": self.X_train.shape[1], "OutputSize": 2},
            "models/regressor.pt")


if __name__ == "__main__":
    features = FeatureExtractor()
    features.prepare_data()

    regression = UseRegression(df=features.df_prepared)
    regression.fit(epochs=1000)
    regression.save()
