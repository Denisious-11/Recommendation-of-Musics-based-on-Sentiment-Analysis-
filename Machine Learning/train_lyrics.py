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
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics import accuracy_score



def print_star():
    print('*'*50, '\n')

#reading dataset
df = pd.read_csv("Project_Dataset/lyrics_dataset.csv")
print("DATA LOADED\n")
print(df.head())
print(df.columns)

print_star()
#######################Preprocessing
#removing unwanted columns
df=df.drop(["file","artist","title","genre","year"], axis=1)

print(df)


#checking null values
print( df.isnull().sum())

#removing rows that contain null values
new_df = df.dropna()
print(new_df)

#checking null values
print( new_df.isnull().sum())


#replace labels with corresponding numerical values
new_df['mood']=new_df['mood'].replace("sad",0)
new_df['mood']=new_df['mood'].replace("happy",1)

#new_df.to_csv('Project_Dataset/final_lyrics_dataset.csv',index=False)

print(new_df)
print_star()

#convert to lowercase
def convert_to_lower(text):
    return text.lower()

new_df['lyrics'] = new_df['lyrics'].apply(lambda x: convert_to_lower(x))

#remove digits
def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

new_df['lyrics'] = new_df['lyrics'].apply(lambda x: remove_numbers(x))

#remove punctuations
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

new_df['lyrics'] = new_df['lyrics'].apply(lambda x: remove_punctuation(x))

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

new_df['lyrics'] = new_df['lyrics'].apply(lambda x: remove_stopwords(x))

#remove_extra_white_spaces
def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

new_df['lyrics'] = new_df['lyrics'].apply(lambda x: remove_extra_white_spaces(x))

#apply lemmatization
def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

new_df['lyrics'] = new_df['lyrics'].apply(lambda x: lemmatizing(x))


# Seperating data and labels
data=new_df["lyrics"]
labels=new_df["mood"]


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
pickle.dump(vectorizer,open('Project_Saved_Models/vectorizer_lyrics.pkl', 'wb'))

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
pickle.dump(clf, open('Project_Saved_Models/l_svm_model.pkl', 'wb'))



# intialize randomforest classifier
clas1 = RandomForestClassifier(n_estimators = 100) 
#Training
clas1.fit(X_train_ros, y_train_ros)
y_pred1 = clas1.predict(X_test_tf)

#calculate accuracy
acc2=accuracy_score(y_test, y_pred1)
print(f"RF Accuracy :{round(acc2,3)*100}%")

#Save the trained model
pickle.dump(clas1, open('Project_Saved_Models/l_rf_model.pkl', 'wb'))


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_ros, y_train_ros)

y_pred2=nb.predict(X_test_tf)


#calculate accuracy
acc3=accuracy_score(y_test, y_pred2)
print(f"NB Accuracy :{round(acc3,3)*100}%")

#Save the trained model
pickle.dump(nb, open('Project_Saved_Models/l_nb_model.pkl', 'wb'))


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train_ros, y_train_ros)

y_pred4 = logreg.predict(X_test_tf)

#calculate accuracy
acc4=accuracy_score(y_test, y_pred4)
print(f"LR Accuracy :{round(acc4,3)*100}%")

#Save the trained model
pickle.dump(logreg, open('Project_Saved_Models/l_lr_model.pkl', 'wb'))




