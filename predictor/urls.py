from django.urls import path

from . import views

urlpatterns = [
    path('predictor', views.predictor,name='predictor'),
    path('signup',views.signupview,name='signup'),
    path('login',views.Login,name='login'),
    path('Tropical_cyclone',views.home,name='home'),
    path('Tropical_cyclone1',views.logout_view),

]