import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import messagebox
from naive_bayes import *
from regression import *
from prepare_features import *
from ffn import *
import torch

label_nb, label_regressor, label_nn = None, None, None


def determine_foreground(y_hat):
    if y_hat == "positive" or y_hat == 1:
        foreground = "green"
    else:
        foreground = "red"

    return foreground


def predict_naive_bayes():
    global label_nb

    nlp = spacy.load("en_core_web_sm")
    with open("models/nb_small.json", "rt", encoding="utf8") as f:
        loaded = json.load(f)

    model = NB(smoothing=0.1)
    model.load(nlp=nlp, label2total=loaded["TotalCounts"], vocab=loaded["Vocab"], naive_bayes=loaded["NaiveBayes"],
               prior_prob=loaded["Priors"])

    text = text_box.get("1.0", tk.END)
    if not text.strip():
        messagebox.showwarning("Bad Input", "No text is read.")
    else:
        y_hat, test_label2prob = model.predict(text.strip())

        if label_nb is not None:
            label_nb.config(text="")

        label_nb = tk.Label(text=y_hat, font=("ArialBold", 15), foreground=determine_foreground(y_hat),
                            background="lightblue")
        label_nb.place(x=35, y=410)


def predict_regression():
    global label_regressor
    loaded = torch.load("models/regressor.pt")
    regressor = LogisticRegression(input_size=loaded["InputSize"], output_size=loaded["OutputSize"])
    regressor.load_state_dict(loaded["STATE"])

    text = text_box.get("1.0", tk.END)
    if not text.strip():
        messagebox.showwarning("Bad Input", "No text is read.")
    else:
        features = FeatureExtractor()
        x = torch.tensor(list(features._to_features(example=text))).float()

        y_hat = np.where(torch.argmax(regressor.forward(x), dim=0).detach().numpy() >= 0.5, 1, 0)

        if label_regressor is not None:
            label_regressor.config(text="")

        label_regressor = tk.Label(text="positive" if y_hat == 1 else "negative", font=("ArialBold", 15),
                                   foreground=determine_foreground(y_hat),
                                   background="lightblue")
        label_regressor.place(x=200, y=410)


def predict_mlp():
    global label_nn

    loaded = torch.load("models/sa.pt", map_location="cpu")
    print(loaded["F1"])
    i, h, o = loaded["Sizes"]
    ffn = FFN(input_size=i, hidden_size=h, output_size=o)
    ffn.load_state_dict(loaded["State"])

    text = text_box.get("1.0", tk.END)
    if not text.strip():
        messagebox.showwarning("Bad Input", "No text is read.")
    else:
        x = process_input(text.strip())
        y_hat = torch.argmax(ffn.forward(x), dim=0)

        if label_nn is not None:
            label_nn.config(text="")

        label_nn = tk.Label(text="positive" if y_hat.item() == 1 else "negative", font=("ArialBold", 15),
                            foreground=determine_foreground(y_hat.item()),
                            background="lightblue")
        label_nn.place(x=400, y=410)


window = tk.Tk()
window.geometry("500x500")
window.title("Sentiment Analysis")
window.resizable(width=False, height=False)
window.config(background="lightblue")

tk.Label(window, text="TWEET", font=("ArialBold", 20), borderwidth=5, foreground="green").place(x=120, y=60)
tk.Label(window, text="SENTIMENT", font=("ArialBold", 20), borderwidth=5, foreground="red").place(x=228, y=60)

text_box = tk.Text(background="white", foreground="black", borderwidth=5, font=("ArialBold", 20), wrap=tk.WORD,
                   width=30, height=5)

tk.Label(text_box, text="PUT YOUR TEXT BELOW", font=("ArialBold", 12), foreground="grey").place(x=130, y=-1)

text_box.insert(tk.END, "\n\n")
text_box.place(x=20, y=150)

tk.Button(window, text="NAIVE BAYES", borderwidth=10, command=predict_naive_bayes).place(x=30, y=350)

tk.Button(window, text="REGRESSION", borderwidth=10, command=predict_regression).place(x=200, y=350)

tk.Button(window, text="SIMPLE NN", borderwidth=10, command=predict_mlp).place(x=370, y=350)

window.mainloop()
