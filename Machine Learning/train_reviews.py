#importing necessary libraries
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier



def print_star():
    print('*'*50, '\n')

#reading dataset
df = pd.read_csv("Project_Dataset/music_album_reviews_dataset.csv")
print("DATA LOADED\n")
print(df.head())
print(df.columns)
df=df[:10000]
print_star()

#######################Preprocessing
#checking null values
print( df.isnull().sum())

#removing rows that contain null values
new_df = df.dropna()
print(new_df)

#checking null values
print( new_df.isnull().sum())

#print(new_df['Rating'].value_counts(ascending=False))

#dropping unnecessary rows
new_df.drop(new_df[new_df['Rating'] == 4.5].index, inplace = True)
new_df.drop(new_df[new_df['Rating'] == 4.0].index, inplace = True)
new_df.drop(new_df[new_df['Rating'] == 3.5].index, inplace = True)
new_df.drop(new_df[new_df['Rating'] == 3.0].index, inplace = True)
new_df.drop(new_df[new_df['Rating'] == 2.5].index, inplace = True)
print(new_df)
#print(new_df['Rating'].value_counts(ascending=False))

#replace labels with corresponding categorical values
new_df['Rating']=new_df['Rating'].replace(5.0,"Positive")
new_df['Rating']=new_df['Rating'].replace(2.0,"Negative")
new_df['Rating']=new_df['Rating'].replace(1.5,"Negative")
new_df['Rating']=new_df['Rating'].replace(1.0,"Negative")
new_df['Rating']=new_df['Rating'].replace(0.5,"Negative")

print(new_df)
#convert categorical values to numerical values
new_df['Rating']=new_df['Rating'].replace("Positive",1)
new_df['Rating']=new_df['Rating'].replace("Negative",0)

print(new_df)
#print(new_df['Rating'].value_counts(ascending=False))

#new_df.to_csv('Project_Dataset/final_music_album_reviews_dataset.csv',index=False)

print_star()

#convert to lowercase
def convert_to_lower(text):
    return text.lower()

new_df['Review'] = new_df['Review'].apply(lambda x: convert_to_lower(x))

#remove digits
def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

new_df['Review'] = new_df['Review'].apply(lambda x: remove_numbers(x))

#remove punctuations
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

new_df['Review'] = new_df['Review'].apply(lambda x: remove_punctuation(x))

#remove stopwords
def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    stop_words.remove('not')
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

new_df['Review'] = new_df['Review'].apply(lambda x: remove_stopwords(x))

#remove_extra_white_spaces
def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

new_df['Review'] = new_df['Review'].apply(lambda x: remove_extra_white_spaces(x))

#apply lemmatization
def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

new_df['Review'] = new_df['Review'].apply(lambda x: lemmatizing(x))


# Seperating data and labels
data=new_df["Review"]
labels=new_df["Rating"]


#Perform train-test splitting
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


print(Counter(y_train))

#Feature Extraction using tf-idf vectorizer
#initialise tf-idf vectorizer
vectorizer = TfidfVectorizer()
#perform feature extraction
vectorizer.fit(X_train)

#train set
X_train_tf = vectorizer.transform(X_train)
X_train_tf = X_train_tf.toarray()
#test set
X_test_tf = vectorizer.transform(X_test)
X_test_tf = X_test_tf.toarray()

#Data balancing
#initialize RandomoverSampler
ROS = RandomOverSampler(sampling_strategy=1)
X_train_ros, y_train_ros = ROS.fit_resample(X_train_tf, y_train)

print(Counter(y_train_ros))

#save vectorizer
pickle.dump(vectorizer,open('Project_Saved_Models/vectorizer_review.pkl', 'wb'))

# import support vector classifier 
# "Support Vector Classifier"
from sklearn.svm import SVC  
clf = SVC(kernel='linear') 
  
# training 
clf.fit(X_train_ros, y_train_ros)
#prediction on test set
y_preds = clf.predict(X_test_tf)

#calculate accuracy
acc1=accuracy_score(y_test, y_preds)
print(f"SVM Accuracy :{round(acc1,3)*100}%")

#Save the trained model
pickle.dump(clf, open('Project_Saved_Models/r_svm_model.pkl', 'wb'))


# intialize randomforest classifier
clas1 = RandomForestClassifier(n_estimators = 100) 
#Training
clas1.fit(X_train_ros, y_train_ros)
y_pred1 = clas1.predict(X_test_tf)

#calculate accuracy
acc2=accuracy_score(y_test, y_pred1)
print(f"RF Accuracy :{round(acc2,3)*100}%")

#Save the trained model
pickle.dump(clas1, open('Project_Saved_Models/r_rf_model.pkl', 'wb'))


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_ros, y_train_ros)

y_pred2=nb.predict(X_test_tf)


#calculate accuracy
acc3=accuracy_score(y_test, y_pred2)
print(f"NB Accuracy :{round(acc3,3)*100}%")

#Save the trained model
pickle.dump(nb, open('Project_Saved_Models/r_nb_model.pkl', 'wb'))


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train_ros, y_train_ros)

y_pred4 = logreg.predict(X_test_tf)

#calculate accuracy
acc4=accuracy_score(y_test, y_pred4)
print(f"LR Accuracy :{round(acc4,3)*100}%")

#Save the trained model
pickle.dump(logreg, open('Project_Saved_Models/r_lr_model.pkl', 'wb'))