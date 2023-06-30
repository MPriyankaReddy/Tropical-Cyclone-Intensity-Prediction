from django.http import HttpResponse
from django.shortcuts import  render, redirect

from django.contrib.auth import login,authenticate
from django.contrib import messages
from .forms import signupform

from django.contrib.auth import logout
from django.shortcuts import redirect


import pickle
with open('./savedModels/model.pkl', 'rb') as f:
    model = pickle.load(f)



def predictor(request):
    if request.method =='POST' :
        lat=float(request.POST['lat'])
        lon=float(request.POST['lon'])
        ecp=float(request.POST['ecp'])
        mssw=float(request.POST['mssw'])
        pd=float(request.POST['pd'])
        sst=float(request.POST['sst'])
        y_pred=model.predict({'lat': lat ,'lon': lon ,'ecp': ecp ,'mssw': mssw ,'pd': pd ,'sst': sst})
        # print(y_pred)
        # print("hello")
        if y_pred==0 :
            y_pred='cyclonic storm'
            return render(request,'cyclonic_storm.html',{'result':y_pred})
        elif y_pred==1 :
            y_pred='Depression'
            return render(request,'depression2.html',{'result':y_pred})
        elif y_pred==2 :
            y_pred='Deep Depression'
            return render(request,'deep_depression.html',{'result':y_pred})
        elif y_pred==3 :
            y_pred='extremely severe cyclonic storm '
            return render(request,'extremly_scs.html',{'result':y_pred})
        elif y_pred==4 :
            y_pred='severe cyclonic storm'
            return render(request,'severe_cs.html',{'result':y_pred})
        elif y_pred==5 :
            y_pred='Super cyclone'
            return render(request,'super_cyclone.html',{'result':y_pred})
        else :
            y_pred='very severe cyclonic storm'
            return render(request,'very_scs.html',{'result':y_pred})

        # return render(request,'main.html',{'result':y_pred})

    return render(request, 'main.html')
 
    

def signupview(request):
    if request.method=='POST' :
        form = signupform(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else :
        form=signupform()
    return render(request,'signup.html',{'form':form})

def Login(request) :
    if request.method=='POST' :
        username=request.POST['username']
        password=request.POST['password']

        user=authenticate(username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('predictor')
        
    else :
        return render(request,'login.html')

def home(request) :
    return render(request,'index.html')


def logout_view(request):
    logout(request)
    return redirect('home')

# Create your views here.
