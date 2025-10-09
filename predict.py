#------------Loading model and predicting------------#



import string
import joblib
from tkinter import *


model= joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#Review Prediction

def predict_review():
    sample_preview = text_box.get(1.0,END)
    sample_preview = sample_preview.lower()
    sample_preview = sample_preview.translate(str.maketrans('','',string.punctuation))
    sample_vec = vectorizer.transform([sample_preview])
    prediction = model.predict(sample_vec)[0]
    confidence = model.predict_proba(sample_vec)[0][1]

    return prediction, confidence

def edit_text():
    prediction, confidence = predict_review()
    if prediction ==1:
            output.config(fg="#00FA9A",text=f"Sentiment: Positive 😃 \n(confidence {round((confidence*100),2)}%)")
    else:
            output.config(fg="#F08080",text=f"Sentiment: Negative 😠 \n(confidence {round((1-confidence)*100,2)}%)")





#Clear

def clear():
    output.config(text="")
    text_box.delete(1.0,END)



window = Tk()
window.title("Sentiment Analysis")
window.config(padx=50, pady=50,bg="#1c1c1c")
window.minsize(height=600,width=600)

#Labels
header = Label(text="Review Sentiment Analyzer 🤖",bg="#1c1c1c",font=("Comfortaa",24,"normal"),fg="#FFF8E7")
header.grid(column=0,row=0,columnspan=3)

output = Label(text="",bg="#1c1c1c",font=("Comfortaa",14,"normal"),fg="#FFF8E7",width=20,height=2)
output.grid(column=1,row=2)

#Text Box
text_box = Text(height=10, width=75,bg="#2a2a2a",fg="#FFF8E7",relief="flat",font=("Comfortaa"))
text_box.focus()
text_box.grid(column=0,row=1,pady=30,columnspan=3)

#Buttons
predict_button = Button(text="Predict",bg="#4169E1",fg="#FFF8E7", activebackground="#5A82F0", activeforeground="#FFF8E7",
                        highlightthickness=0 ,relief="flat",cursor="hand2",width=25,height=2,font=("Comfortaa",14,"bold"),command=edit_text)
predict_button.grid(column=0,row=3)

clear_button = Button(text="Clear",bg="#CD5C5C",fg="#FFF8E7",activebackground="#E06A6A", activeforeground="#FFF8E7",
                      highlightthickness=0,relief="flat",cursor="hand2",width=25,height=2,font=("Comfortaa",14,"bold"),command=clear)
clear_button.grid(column=2,row=3,pady=50)

window.mainloop()







