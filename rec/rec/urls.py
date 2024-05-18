"""rec URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from rec_app.views import *

urlpatterns = [
    url(r'^admin/', admin.site.urls),

    ##########login & registration start
    url(r'^$',display_login),
    url(r'^show_register',show_register,name="show_register"),
    url(r'^register', register, name="register"),
    url(r'^display_login', display_login, name="display_login"),
    url(r'^check_login', check_login, name="check_login"),
    url(r'^logout',logout,name="logout"),
    ##########login & registration end


    ################Admin start
    url(r'^show_home_user',show_home_user,name="show_home_user"),
    url(r'^reviews_sa',reviews_sa,name="reviews_sa"),
    url(r'^lyrics_sa',lyrics_sa,name="lyrics_sa"),
    url(r'^predict_review',predict_review,name="predict_review"),
    url(r'^predict_lyrics',predict_lyrics,name="predict_lyrics"),
    url(r'^display_recommendations',display_recommendations,name="display_recommendations"),
    url(r'^get_recommend',get_recommend,name="get_recommend"),
    url(r'^d_recommend_review',d_recommend_review,name="d_recommend_review"),
    url(r'^n_recommend_review',n_recommend_review,name="n_recommend_review"),
    ################Admin end
]
