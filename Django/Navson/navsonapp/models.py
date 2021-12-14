from django.db import models

# Create your models here.
class User(models.Model):
    Username=models.CharField(max_length=200)
    Email=models.EmailField(max_length=200)
    Password=models.CharField(max_length=100)
class Users(models.Model):
    username=models.CharField(max_length=100)
    date=models.DateField()
    exercise=models.CharField(max_length=100)
    on_time=models.TimeField()
    end_time=models.TimeField()
    workout_time=models.CharField(max_length=100)

