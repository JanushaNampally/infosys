from django.urls import path, re_path
from . import views

urlpatterns = [

    path('', views.home, name='home'),

    path('login/', views.login_view, name='login'),

    path('register/', views.register, name='register'),

    path('logout/', views.logout_view, name='logout'),

    path('monitor/', views.monitor, name='monitor'),

    path('history/', views.history, name='history'),

    path('profile/', views.profile, name='profile'),

    path('settings/', views.settings_page, name='settings'),

    re_path(r'^stream/(?P<path>.+)$', views.stream_media, name='stream_media'),

]