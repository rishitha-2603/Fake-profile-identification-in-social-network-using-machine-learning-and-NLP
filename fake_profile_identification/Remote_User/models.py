from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class profile_identification_type(models.Model):

    prof_idno= models.CharField(max_length=3000)
    name= models.CharField(max_length=3000)
    screen_name= models.CharField(max_length=3000)
    statuses_count= models.CharField(max_length=3000)
    followers_count= models.CharField(max_length=3000)
    friends_count= models.CharField(max_length=3000)
    created_at= models.CharField(max_length=3000)
    location= models.CharField(max_length=3000)
    default_profile= models.CharField(max_length=3000)
    prf_image_url= models.CharField(max_length=3000)
    prf_banner_url= models.CharField(max_length=3000)
    prf_bgimg_https= models.CharField(max_length=3000)
    prf_text_color= models.CharField(max_length=3000)
    profile_image_url_https= models.CharField(max_length=3000)
    prf_bg_title= models.CharField(max_length=3000)
    profile_background_image_url= models.CharField(max_length=3000)
    description= models.CharField(max_length=3000)
    Prf_updated= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


