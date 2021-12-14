from django.urls import path
from . import views
urlpatterns=[
    path("",views.login,name="login"),
    path("home",views.home,name="home"),
    path("profile",views.profile,name="profile"),
    path("history",views.history,name="workouthistory"),
    path("startworkout",views.startwork,name="startwork"),
    path("start",views.start,name="start"),
    path("end",views.end,name="end"),
    path("loginform",views.loginform,name="loginform"),
    path("signupform",views.signupform,name="signupform"),
    path("signup",views.signup,name="signup"),
    path("biceppage",views.biceppage,name="biceppage"),
    path("bicep",views.bicep,name="bicep"),
    path("shoulder_shrugpage",views.shoulder_shrugpage,name="shoulder_shrugpage"),
    path("shoulder_shrug",views.shoulder_shrug,name="shoulder_shrug"),
    path("squat",views.squat,name="squat"),
    path("squatpage",views.squatpage,name="squatpage"),
    path("yoga",views.yoga,name="yoga"),
    path("yogapage",views.yogapage,name="yogapage"),
]