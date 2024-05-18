#importing necessary libraries
import numpy as np
import tkinter as tk
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')#tokenize the text in the dataset.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import string
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics import accuracy_score

print("\nSystem Testing...")
#Load the trained model & Vectorizer
loaded_model = pickle.load(open('Project_Saved_Models/model_lyrics.pkl', 'rb'))
vectorizer = pickle.load(open('Project_Saved_Models/vectorizer_lyrics.pkl', 'rb'))

#Preprocessing
def convert_to_lower(text):
    return text.lower()

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    stop_words.remove('not')
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)


###########################

root= tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=300)
canvas1.pack()

entry1 = tk.Entry(root) 
canvas1.create_window(200, 140, window=entry1)


def predict_lyrics(lyrics):


    a=convert_to_lower(lyrics)
    b=remove_numbers(a)
    c=remove_punctuation(b)
    d=remove_stopwords(c)
    e=remove_extra_white_spaces(d)
    f=lemmatizing(e)


    X_test = vectorizer.transform([f])

    X_test = X_test.toarray()

    result=loaded_model.predict(X_test)
    print(result)
    final_result=result[0]
    print("\n")
    print("Sentiment Analysis Result\n")
    if final_result==0:
        print("Sad")
    if final_result==1:
        print("Happy")


def pred():  
    x1 = entry1.get()

    print(x1)
    print(type(x1))

    predict_lyrics(x1)
    
    
button1 = tk.Button(text='Predict', command=pred)
canvas1.create_window(200, 180, window=button1)

root.mainloop()