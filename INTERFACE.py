import os
import tkinter as tk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model import *

label_positive, label_neutral, label_negative, label_prediction = None, None, None, None


def clear_division(string):
    if 'e' in string:
        p1, p2 = string.split('.')
        string = f"{p1}.{p2[: 7]}*10^{-int(p2[p2.index('e') + 2:])}"
    else:
        string = string[: 7]

    return string


def eval_sentiment(t):
    global label_positive, label_neutral, label_negative, label_prediction

    if label_positive is not None:
        label_positive.config(text="0.0")
    if label_neutral is not None:
        label_neutral.config(text="0.0")
    if label_negative is not None:
        label_negative.config(text="0.0")
    if label_prediction is not None:
        label_prediction.config(text="None")

    if t == "small":
        filename = "models/nb_small.json"
    else:
        filename = "models/nb_large.json"

    with open(filename, "rt", encoding="utf8") as f:
        model = json.load(f)

    nlp = spacy.load("en_core_web_sm")
    my_model = NB()
    my_model.load(nlp=nlp, label2total=model["TotalCounts"], vocab=model["Vocab"], naive_bayes=model["NaiveBayes"],
                  prior_prob=model["Priors"])
    text = text_box.get(1.0, "end")

    text = "".join(text.split('\n')[1:])
    _, predictions = my_model.predict(text)

    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    for lab, prob in predictions:
        if not text.strip():
            probs[lab] = 0.0
        else:
            probs[lab] = prob

    pos, neg, neu = str(probs['positive']), str(probs['negative']), str(probs['neutral'])

    label_positive = tk.Label(text=f"positive\n{clear_division(pos)}", font=("ArialBold", 10), foreground="green")
    label_positive.place(x=80, y=370)

    label_negative = tk.Label(text=f"negative\n{clear_division(neg)}", font=("ArialBold", 10), foreground="red")
    label_negative.place(x=200, y=370)

    label_neutral = tk.Label(text=f"neutral\n{clear_division(neu)}", font=("ArialBold", 10), foreground="grey")
    label_neutral.place(x=320, y=370)

    y_hat = sorted(probs.items(), key=lambda x: x[1], reverse=True)[0][0]

    if y_hat == "positive":
        foreground = "green"
    elif y_hat == "negative":
        foreground = "red"
    else:
        foreground = "grey"
    label_prediction = tk.Label(text=f"Prediction: {y_hat}", font=("ArialBold", 15), foreground=foreground)
    label_prediction.place(x=160, y=430)


window = tk.Tk()
window.geometry("500x500")
window.title("Naive Bayes")
window.resizable(width=False, height=False)
window.config(background="lightblue")

main_frame = tk.Frame(window)
main_frame.pack()

label = tk.Label(main_frame, text="TWEET SENTIMENT", font=("TimesNewRoman", 20), foreground="blue")
label.place(x=80, y=60)

text_box = tk.Text(main_frame, background="black", foreground="blue", width=50, height=10)
text_box.insert(tk.END, "\t\tPUT YOUR TEXT BELOW\n")
text_box.pack(pady=150)

button1 = tk.Button(main_frame, text="NB SMALL", borderwidth=5, command=lambda t="small": eval_sentiment(t))
button1.place(x=95, y=320)

button2 = tk.Button(main_frame, text="NB LARGE", borderwidth=5, command=lambda t="large": eval_sentiment(t))
button2.place(x=250, y=320)

window.mainloop()
