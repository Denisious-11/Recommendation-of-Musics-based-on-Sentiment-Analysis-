from django.shortcuts import render
import json
from django.core import serializers
from .models import *
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.db.models import Count
import re
from django.views.decorators.cache import never_cache
from django.core.files.storage import FileSystemStorage
import base64
import cv2
import os
import numpy as np
import random
from datetime import datetime
from datetime import date
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')#tokenize the text in the dataset.
nltk.download('stopwords')

# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
import string

#Load the trained model & Vectorizer
loaded_model1 = pickle.load(open('rec_app/static/trained_models/model_review.pkl', 'rb'))
vectorizer1 = pickle.load(open('rec_app/static/trained_models/vectorizer_review.pkl', 'rb'))
loaded_model2 = pickle.load(open('rec_app/static/trained_models/model_lyrics.pkl', 'rb'))
vectorizer2 = pickle.load(open('rec_app/static/trained_models/vectorizer_lyrics.pkl', 'rb'))


# Create your views here.
@never_cache
###############LOGIN & REGISTRATION START
def display_login(request):
    return render(request, "login.html", {})


def show_register(request):
    return render(request, "register.html", {})

@never_cache
def logout(request):
    if 'uid' in request.session:
        del request.session['uid']
    return render(request,'login.html')


def register(request):
	
	username = request.GET.get("uname")
	phone = request.GET.get("phone")
	email=request.GET.get("email_id")
	password = request.GET.get("pass")
	age=request.GET.get("age")
	gender=request.GET.get("gender")

	a = Users.objects.filter(username=username)
	c = a.count()
	if(c == 1):
	    return HttpResponse("[INFO]: Username is already Taken, Choose another one")
	else:
		if re.match(r'^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$',email):

			obj1=Users.objects.filter(username=username, phone=phone, email=email,password=password,age=age,gender=gender)
			cc=obj1.count()

			if(cc==1):
				return HttpResponse("Already Registered")
			else:

			    b = Users(username=username, phone=phone, email=email,password=password,age=age,gender=gender)
			    b.save()

			    return HttpResponse("Registration Successful")
		else:
			return HttpResponse("Try valid email id")


def check_login(request):
	username = request.GET.get("uname")
	password = request.GET.get("password")

	print(username)
	print(password)



	d = Users.objects.filter(username=username, password=password)
	c = d.count()
	if c == 1:
		d2 = Users.objects.get(username=username, password=password)
		request.session["uid"] = d2.u_id
		request.session["username"]=d2.username
		return HttpResponse("Login Successful")
	else:
		return HttpResponse("Invalid")

###############LOGIN & REGISTRATION END

@never_cache
def show_home_user(request):
	if 'uid' in request.session:
		u_id=request.session['uid']
		obj1=Users.objects.get(u_id=int(u_id))
		u_name=obj1.username
		return render(request,'home_user.html',{'username':u_name}) 
	else:
		return render(request,'login.html')


@never_cache
def reviews_sa(request):
	if 'uid' in request.session:
		return render(request,'reviews_user.html',) 
	else:
		return render(request,'login.html')


@never_cache
def lyrics_sa(request):
	if 'uid' in request.session:
		return render(request,'lyrics_user.html',) 
	else:
		return render(request,'login.html')



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


def predict_review(request):
	song_name=request.POST.get("song_name")
	username=request.session['username']
	review=request.POST.get("review")
	print(review)
	# print(type(review))
	a=convert_to_lower(review)
	b=remove_numbers(a)
	c=remove_punctuation(b)
	d=remove_stopwords(c)
	e=remove_extra_white_spaces(d)
	f=lemmatizing(e)


	X_test = vectorizer1.transform([f])

	X_test = X_test.toarray()

	result=loaded_model1.predict(X_test)
	print(result)
	final_result=result[0]
	print("\n")
	print("Sentiment Analysis Result\n")
	if final_result==0:
		print("Negative")
		obj2=Recommend(username=username,song_name=song_name,review_sentiment="Negative",review=review)
		obj2.save()
		return HttpResponse("<script>alert('Negative Review');window.location.href='/reviews_sa/'</script>")

	if final_result==1:
		print("Positive")

		obj1=Recommend(username=username,song_name=song_name,review_sentiment="Positive",review=review)
		obj1.save()
		return HttpResponse("<script>alert('Positive Review');window.location.href='/reviews_sa/'</script>")

def predict_lyrics(request):
	lyrics=request.POST.get("lyrics")
	print(lyrics)
	# print(type(lyrics))
	a=convert_to_lower(lyrics)
	b=remove_numbers(a)
	c=remove_punctuation(b)
	d=remove_stopwords(c)
	e=remove_extra_white_spaces(d)
	f=lemmatizing(e)


	X_test = vectorizer2.transform([f])

	X_test = X_test.toarray()

	result=loaded_model2.predict(X_test)
	print(result)
	final_result=result[0]
	print("\n")
	print("Sentiment Analysis Result\n")
	if final_result==0:
		print("Sad")
		return HttpResponse("<script>alert('Predicted Mood : Sad');window.location.href='/lyrics_sa/'</script>")

	if final_result==1:
		print("Happy")
		return HttpResponse("<script>alert('Predicted Mood : Happy');window.location.href='/lyrics_sa/'</script>")

@never_cache
def display_recommendations(request):
	if 'uid' in request.session:
		return render(request,'get_recommendation.html',) 
	else:
		return render(request,'login.html')


@never_cache
def d_recommend_review(request):
	if 'uid' in request.session:
		my_u_name=request.session["username"]
		obj1= Recommend.objects.all().filter(review_sentiment="Positive").exclude(username=my_u_name).values('song_name','review','username').distinct()
		print(obj1)
		return render(request,"recommendations_review.html",{'data':obj1})
	else:
		return render(request,'login.html')

@never_cache
def n_recommend_review(request):
	if 'uid' in request.session:
		my_u_name=request.session["username"]
		obj1=Recommend.objects.all().filter(review_sentiment="Negative").exclude(username=my_u_name).values('song_name','review','username').distinct()
		print("************************")
		print("************************")
		print(obj1)
		return render(request,"recommendations_review.html",{'data':obj1})
	else:
		return render(request,'login.html')

def get_recommendations(emotion,df):
	NUM_RECOMMEND=5
	# happy_set=[]
	# sad_set=[]
	if emotion=="Happy":
		get_lists1=df[df['kmeans']==0]['song_name']
		# print(get_lists1)
		shuffled = get_lists1.sample(frac = 1)
		# print(shuffled)
		happy_set=shuffled.head(NUM_RECOMMEND)
		# print("*******************************")
		# print(happy_set)
		happy_set=happy_set.reset_index(drop=True)
		# print("^^^^^^^^^^^^^^^^^^^^^^^^")
		# print(happy_set)
		# print(type(happy_set))
		happy_set = happy_set.tolist()
		
		return happy_set
	
	else:
		get_lists2=df[df['kmeans']==1]['song_name']
		# print(get_lists2)
		shuffled1 = get_lists2.sample(frac = 1)
		# print(shuffled1)
		sad_set=shuffled1.head(NUM_RECOMMEND)
		# print("*******************************")
		# print(sad_set)
		sad_set=sad_set.reset_index(drop=True)
		# print("^^^^^^^^^^^^^^^^^^^^^^^^")
		# print(sad_set)
		sad_set = sad_set.tolist()

		return sad_set


def get_recommend(request):
	selected_mood=request.POST.get("mood")
	print(selected_mood)
	if selected_mood=="Select":
		return HttpResponse("<script>alert('Please Select a Mood');window.location.href='/display_recommendations/'</script>")
	else:

		#read the dataset (based on popularity { users liked })
		data=pd.read_csv('rec_app/static/data.csv.zip',compression='zip')
		# print(data)
		# print(data.columns)

		#remove duplicate song names
		data.drop_duplicates(inplace=True,subset=['name'])
		name=data['name']
		# print(data)

		#select features for clustering
		col_features = ['danceability', 'energy', 'valence', 'loudness']
		#perform feature scaling
		X = MinMaxScaler().fit_transform(data[col_features])
		#kmeans initialization
		kmeans = KMeans(init="k-means++",
		                n_clusters=2,
		                random_state=15).fit(X)
		#add new columns to dataframe
		data['kmeans'] = kmeans.labels_
		data['song_name']=name
		# print(data)
		# print(data.columns)

		#clustering
		cluster=data.groupby(by=data['kmeans'])

		#sorting based on popularity
		df=cluster.apply(lambda x: x.sort_values(["popularity"],ascending=False))
		# print(df)
		#df.reset_index(level=0, inplace=True)
		df.reset_index(drop=True)


		result=get_recommendations(selected_mood,df)
		print("####################")
		print(result)
		print(type(result))
		# print(result[0])
		

		return render(request,"my_recommendations.html",{'data':result})