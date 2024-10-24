import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import spacy
from gensim.models import fasttext
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

nlp, embeddings = spacy.load("en_core_web_sm"), None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        return self.linear2(self.relu(self.linear1(X)))


def process_input(text):
    global nlp, embeddings
    if embeddings is None:
        embeddings = fasttext.KeyedVectors.load_word2vec_format("small_data/crawl-300d-2M.vec", binary=False)

    assert nlp is not None, "spaCY not loaded."

    tokens = nlp(text)
    representation, count = torch.zeros((300,)), 0
    for token in tokens:
        if not token.is_stop and not token.is_punct and not token.like_num and not token.like_url and not token.like_url:
            w = token.text.lower().strip(string.punctuation).strip()
            if w in embeddings:
                representation += embeddings[w]
                count += 1
    if count == 0: count = 1
    representation /= count
    representation = torch.tensor(representation, dtype=torch.float32)

    return representation


def to_embeddings(train=True, polarity="pos"):
    if train:
        path = f"small_data/train"
    else:
        path = f"small_data/test"
    examples = os.listdir(path)
    e = []
    for f in examples:
        with open(f"{path}/{polarity}/{f}", "rt", encoding="utf8") as reader:
            e.append(process_input(reader.read()))
    e = torch.stack([r for r in e])
    torch.save({"TextMeanEmbedding": e, "Label": 0 if polarity == "neg" else 1}, f"{path}/{polarity}.pt")


def train_and_save():
    global device
    e_p1, e_n1 = torch.load("small_data/train/pos.pt")["TextMeanEmbedding"], \
        torch.load("small_data/train/neg.pt")["TextMeanEmbedding"]
    e_p2, e_n2 = torch.load("small_data/test/pos.pt")["TextMeanEmbedding"], \
        torch.load("small_data/test/neg.pt")["TextMeanEmbedding"]

    X = torch.cat([e_p1, e_p2, e_n1, e_n2], dim=0).float().to(device)
    Y = torch.cat(
        [torch.ones((e_p1.shape[0] + e_p2.shape[0],)), torch.zeros((e_n1.shape[0] + e_n2.shape[0],))]).long().to(device)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, test_size=.2)

    ffn = FFN(input_size=X_train.shape[1], hidden_size=64, output_size=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ffn.parameters(), lr=0.001)

    epochs = 1000

    loss_final = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(ffn.forward(X_train), Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0: loss_final = loss.item()

    y_hat = torch.argmax(ffn.forward(X_test), dim=1).cpu()
    Y_test = Y_test.cpu()
    f1 = f1_score(Y_test, y_hat, average="macro")

    torch.save(
        {"State": ffn.state_dict(), "Sizes": (X_train.shape[1], 64, 2), "Epochs": 1000, "Loss": loss_final, "F1": f1},
        "models/sa.pt")


if __name__ == "__main__":
    # to_embeddings(train=True, polarity="pos")
    # to_embeddings(train=True, polarity="neg")
    # to_embeddings(train=False, polarity="pos")
    # to_embeddings(train=False, polarity="neg")

    train_and_save()
