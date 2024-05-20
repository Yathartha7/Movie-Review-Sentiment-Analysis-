import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_bert import get_custom_objects

features = pickle.load(open('tf_model.preproc', 'rb'))
model = keras.models.load_model('tf_model.h5', custom_objects=get_custom_objects())

positive_count = 0
negative_count = 0

root = tk.Tk()
root.title("Movie Review Sentiment Analysis")
root.attributes('-fullscreen', True)  


def predict_sentiment():
    review = review_text.get("1.0", tk.END).strip()  

    if len(review) > 0:
        preprocessed_data = features.preprocess([review])
        result = model.predict(preprocessed_data)

    
        sentiment = "Negative" if result[0][0] > 0.5 else "Positive"

        
        result_label.config(text=f"Predicted Sentiment: {sentiment}")

        
        result_label.after(3000, lambda: result_label.config(text=""))

    
        if sentiment == "Positive":
            global positive_count
            positive_count += 1
        else:
            global negative_count
            negative_count += 1

       
        positive_count_label.config(text=f"Positive Reviews: {positive_count}")
        positive_count_label.config(style="Positive.TLabel")
        negative_count_label.config(text=f"Negative Reviews: {negative_count}")
        negative_count_label.config(style="Negative.TLabel")
        root.update()

       
        total_reviews = positive_count + negative_count
        rating = int((positive_count / total_reviews) * 10) if total_reviews > 0 else 0
        rating_label.config(text=f"Movie Rating: {rating}/10")

    else:
        messagebox.showwarning("Input Error", "Please enter a movie review.")


def reset_parameters():
    global positive_count, negative_count
    positive_count = 0
    negative_count = 0
    positive_count_label.config(text="Positive Reviews: 0")
    negative_count_label.config(text="Negative Reviews: 0")
    rating_label.config(text="Movie Rating: 0/10")
    result_label.config(text="")


def clear_text():
    review_text.delete("1.0", tk.END)


def exit_application():
    root.quit()


style = ttk.Style()


style.configure("Positive.TLabel", foreground="green", font=("Helvetica", 24, "bold"))

style.configure("Negative.TLabel", foreground="red", font=("Helvetica", 24, "bold"))

header_label = ttk.Label(root, text="Movie Review Sentiment Analysis", font=("Arial", 50, "bold"))
header_label.pack(pady=(100, 10))  

frame = ttk.Frame(root)
frame.pack(expand=True, pady=50)

review_label = ttk.Label(frame, text="Enter a movie review:", font=("Helvetica", 24))
review_label.pack()

review_text = tk.Text(frame, height=5, width=40, font=("Helvetica", 24), highlightthickness=1, highlightbackground="black")  # Adjust the height of the text box
review_text.pack(pady=10)  

buttons_frame = ttk.Frame(frame)
buttons_frame.pack(pady=20)

style.configure("Custom.TButton", font=("Helvetica", 24), width=15)  

predict_button = ttk.Button(buttons_frame, text="Predict Sentiment", command=predict_sentiment, style="Custom.TButton")
predict_button.pack(side="left", padx=10)

reset_button = ttk.Button(buttons_frame, text="Reset", command=reset_parameters, style="Custom.TButton")
reset_button.pack(side="left", padx=10)

clear_button = ttk.Button(buttons_frame, text="Clear", command=clear_text, style="Custom.TButton")
clear_button.pack(side="left", padx=10)

exit_button = ttk.Button(buttons_frame, text="Exit", command=exit_application, style="Custom.TButton")
exit_button.pack(side="left", padx=10)

result_label = ttk.Label(frame, text="", font=("Helvetica", 24))
result_label.pack()

positive_count_label = ttk.Label(frame, text="Positive Reviews: 0", font=("Helvetica", 24))
positive_count_label.pack()

negative_count_label = ttk.Label(frame, text="Negative Reviews: 0", font=("Helvetica", 24))
negative_count_label.pack()

rating_label = ttk.Label(frame, text="Movie Rating: 0/10", font=("Helvetica", 24))
rating_label.pack()

root.mainloop()