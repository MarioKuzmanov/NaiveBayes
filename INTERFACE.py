import os
import tkinter as tk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model import *

label_positive, label_neutral, label_negative, label_prediction = None, None, None, None


window = tk.Tk()
window.geometry("500x500")
window.title("Naive Bayes")
window.resizable(width=False, height=False)
window.config(background="lightblue")

main_frame = tk.Frame(window)
main_frame.pack()

label = tk.Label(main_frame, text="TWEET SENTIMENT", font=("TimesNewRoman", 20), foreground="blue")
label.place(x=80, y=60)

text_box = tk.Text(main_frame, background="black", foreground="blue", font=("ArialBold", 20), width=50, height=10)
text_box.insert(tk.END, "\tPUT YOUR TEXT BELOW\n")
text_box.pack(pady=150)

button1 = tk.Button(main_frame, text="NAIVE BAYES", borderwidth=5, command=lambda t="small": eval_sentiment(t))
button1.place(x=95, y=320)


window.mainloop()
