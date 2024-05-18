from django.db import models

# Create your models here.
class Users(models.Model):
    u_id = models.IntegerField(primary_key=True)
    username = models.CharField(max_length=255)
    phone = models.CharField(max_length=255)
    email=models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    age=models.CharField(max_length=255)
    gender=models.CharField(max_length=255)


class Recommend(models.Model):
    r_id= models.IntegerField(primary_key=True)
    username = models.CharField(max_length=255)
    song_name = models.CharField(max_length=255)
    review_sentiment=models.CharField(max_length=255)
    review=models.TextField()
 