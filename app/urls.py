# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views

urlpatterns = [
    # Matches any html file 
    re_path(r'^.*\.html', views.pages, name='pages'),

    # The home page
    path('', views.index, name='home'),
    path('map', views.map, name='map'),
    path('wilayas', views.wilayas, name='wilayas'),
    path('evolution', views.evolution, name='evolution'),
    path('generate', views.auto_arima, name='auto_arima'),
    path('api/countries/Algeria', views.covid_data, name='countries'),
]
