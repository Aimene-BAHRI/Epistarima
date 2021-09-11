# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template

import json
import requests
from core.settings import MAPBOX_TOKEN
from django.http import JsonResponse 

import pandas as pd
def covid_data(request):
	country = 'Algeria'

	# get country info with it's geoJSON
	countries_file = open('core/geoJson/Countries.json', 'r')
	# covid/static/geoJson/Countries/Algeria/Algeria_48.Json
	countries_json = json.load(countries_file)

	Country = countries_json[country]

	with open('core/geoJson/Countries/{}/{}_48.Json'.format(country, country), 'r') as f:
		country_base_geoJson = json.loads(f.read())

		Country['geoJSON'] = country_base_geoJson
	covid_data = requests.get('https://api.corona-dz.live/province/latest').json()
	unhundled_cities = []
	for city in Country['geoJSON']:
		for city2 in covid_data:
			if city == city2['name']:
				Country['geoJSON'][city]['properties']['density'] = city2['data'][0]
				pass
			else:
				unhundled_cities.append({
					'city_name': city,
					'city_id': Country['geoJSON'][city]['properties']['id'],
				})
	context = {
		'Country' : Country,
		'unhundled_cities' : unhundled_cities
	}
	
	return JsonResponse(context)

@login_required(login_url="/login/")
def index(request):
    api = requests.get('https://api.corona-dz.live/country/summary').json()
    context = {
        'api': api
    }
    return render(request, "index.html", context)

@login_required(login_url="/login/")
def map(request):
    
    api = requests.get('https://api.corona-dz.live/country/summary').json()
    context = {
        'api': api,
        'MAPBOX_TOKEN': MAPBOX_TOKEN,
    }
    return render(request, "map.html", context)


@login_required(login_url='/login/')
def wilayas(request):
    api = requests.get('https://api.corona-dz.live/province/latest').json()
    
    context = {
        'api': api,
    }
    return render(request, "wilayas.html", context)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

@login_required(login_url='/login/')
def auto_arima(request):
    import pandas as pd
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    
    all_data = requests.get('https://api.corona-dz.live/country/all').json()
    json_data = json.dumps(all_data)
    
    df = pd.read_json(json_data)
    df.to_csv('data.csv',)
    data = pd.read_csv('data.csv', parse_dates=True)
    data =data.dropna()
    data['confirmed'] = data['confirmed'].astype('float32')
    data['recovered'] = data['recovered'].astype('float32')
    data['deaths'] = data['deaths'].astype('float32')
    data['date'] = pd.to_datetime(data['date'])
    data['updatedAt'] = pd.to_datetime(data['updatedAt'])

    data.set_index('date', inplace=True)
    from pmdarima.arima import auto_arima

    train = data['confirmed'][:int(len(df)*(80/100))]
    test = data['confirmed'][-int(len(df)*(20/100)):]
    print(len(test))
    arima_model = None
    graphic = None
    score = None
    if request.method == 'POST': 
        arima_model =  auto_arima(
            train,
            test='adf',
            start_p=1, d=None, start_q=1, 
            max_p=3, max_q=3, 
            start_P=0, D=0, start_Q=0,  
            m=1, seasonal=False, 
            trace = True,
            error_action='ignore',
            supress_warnings=True,
            stepwise = True)
        print('lol\n', arima_model.summary())

        # Forecast
        n_periods = int(len(df)*(20/100))
        fc, confint = arima_model.predict(n_periods=n_periods, return_conf_int=True)
        # index_of_fc = np.arange(len(train), len(train)+n_periods)
        # make series for plotting purpose
        fc_series = pd.Series(fc, index=test.index)
        lower_series = pd.Series(confint[:, 0], index=test.index)
        upper_series = pd.Series(confint[:, 1], index=test.index)
        fig1 = plt.figure()
        # Plot

        plt.plot(train)
        plt.plot(fc_series, color='darkgreen')
        plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
        plt.title("Final Forecast of confirmed data")
        plt.legend()
        fig1.savefig('arima/train_test_predict.png')
        import base64
        from io import BytesIO

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')

        # from sklearn.metrics import r2_score
        # test['predicted_confirmed'] = prediction
        # print(len(test))
        # score = r2_score(data['confirmed'][-int(len(df)*(20/100)):], test['predicted_confirmed'])

    context = {
        'api': data,
        'best_model': arima_model,
        'graphic': graphic,
        "score" : score
    }
    return render(request, "generate_auto_arima.html", context)

@login_required(login_url='/login/')
def evolution(request):
    all_data = requests.get('https://api.corona-dz.live/province/latest').json()
    json_data = json.dumps(all_data)
    adrar = requests.get('https://api.corona-dz.live/province/1/all').json()
    adrar_json = json.dumps(adrar)
    df = pd.read_json(json_data)
    adrar_dataframe = pd.read_json(adrar_json)
    
    provinces_names = []
    provinces_confirmed_data = []
    provinces_confirmed_date = []

    for province in range(df.shape[0]):
        provinces_names.append([df["provinceId"][province], df["name"][province]])
    
    
    
        

    for data in adrar_dataframe["data"][0] :
        print(data['date'])
        provinces_confirmed_data.append(data["confirmed"])
        provinces_confirmed_date.append(data['date'])    

    context = {
        'names': provinces_names,
        'provinces_confirmed_data' : provinces_confirmed_data,
        'dates' : provinces_confirmed_date
    }
    return render(request, "evolution.html", context)


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template = request.path.split('/')[-1]
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))
        
    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'error-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'error-500.html' )
        return HttpResponse(html_template.render(context, request))
