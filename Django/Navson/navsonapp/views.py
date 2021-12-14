from django.http.response import HttpResponse
from django.shortcuts import render
from navsonapp.exercise import Bicep,Shoulder_Shrug,Squat,Yoga,err
from .models import User,Users
from datetime import datetime, date
from math import sqrt
from django.http.response import StreamingHttpResponse
# Create your views here.
def login(request):
    return render(request,"login.html")
def home(request):
    return render(request,"home.html")
def startwork(request):
    return render(request,"menupage.html")
def profile(request):
    data=User.objects.all()
    return render(request,"profile.html",{"data":data})
def loginform(request):
    name=request.POST["username"]
    request.session['name']=name
    return render(request,"next_to_login.html")
def signup(request):
    return render(request,"signup.html")
def signupform(request):
    name=request.POST["username"]
    email=request.POST["email"]
    password=request.POST["password"]
    details=User(Username=name,Email=email,Password=password)
    details.save()
    return render(request,"login.html")
def history(request):
    datas=Users.objects.all()
    username=request.session['name']
    return render(request,"workout_history.html",{"data":datas,"name":[username]})
def start(request):
    global start_time
    start_time=datetime.now()
    return render(request,"bicep.html")
def end(request):
    end_time=datetime.now()
    net_time= (end_time)-(start_time)
    username=request.session['name']
    exercise_name=request.session['exercise_name']
    info=Users(username=username,date=date.today(),exercise=exercise_name,on_time=start_time,end_time=end_time,workout_time=net_time)
    info.save()
    return render(request,"bicep.html")
def biceppage(request):
    request.session['exercise_name']="Bicep"
    return render(request,"bicep.html")
def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def bicep(request):
	return StreamingHttpResponse(gen(Bicep()),
					content_type='multipart/x-mixed-replace; boundary=frame')
def shoulder_shrugpage(request):
    request.session['exercise_name']="Shoulder_Shrug"
    return render(request,"shoulder_shrug.html")
def shoulder_shrug(request):
	return StreamingHttpResponse(gen(Shoulder_Shrug()),
					content_type='multipart/x-mixed-replace; boundary=frame')  
def squatpage(request):
    request.session['exercise_name']="Squat"
    return render(request,"squat.html",{"answer":err})
def squat(request):
	return StreamingHttpResponse(gen(Squat()),
					content_type='multipart/x-mixed-replace; boundary=frame')
def yogapage(request):
    request.session['exercise_name']="Yoga"
    return render(request,"yoga.html")
def yoga(request):
	return StreamingHttpResponse(gen(Yoga()),
					content_type='multipart/x-mixed-replace; boundary=frame')