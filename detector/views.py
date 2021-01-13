from django.shortcuts import render
from .apps import DetectorConfig
import pandas as pd
from pandas import Series
import random
import numpy as np
from sklearn.metrics import f1_score
#import time
#import schedule
#from celery.schedules import crontab
#from celery.task import periodic_task
#from datetime import timedelta

# Create your views here.
#@periodic_task(run_every=crontab(hour=7, minute=30, day_of_week="mon"))
#@periodic_task(run_every=timedelta(seconds=3))
def home(request):


	#def run_results():
	#	data = DetectorConfig.data
	#	user_input = data.sample(n=1)



	#user_input = schedule.every(3).seconds.do(run_results)

	#print(user_input)


	

	data = DetectorConfig.data

	user_input = data.sample(n=1)

	transact_id = user_input.index
	

	y = user_input['Class']
	x = user_input.drop(columns=['Class'], inplace=False) 

	data_prep = DetectorConfig.data_prep_pipe.transform(x)



	#prediction = DetectorConfig.model.predict(data_prep);
	#predict_probability = DetectorConfig.model.predict_proba(data_prep);
	prediction = DetectorConfig.calibration.predict(data_prep);
	predict_probability = DetectorConfig.calibration.predict_proba(data_prep);


	# defining a function to calculate cost savings
	def cost_saving(ytrue, ypred, amount, threshold=0.5,admin_cost=2.5, epsilon=1e-7):
		ypred = ypred.flatten()
		fp = np.sum((ytrue == 0) & (ypred == 1))
		cost = np.sum(fp*admin_cost) + np.sum((amount[(ytrue == 1) & (ypred == 0)]))
		max_cost = np.sum((amount[(ytrue == 1)])) 
		savings = 1 - (cost/(max_cost+epsilon))

		return savings

	#is_fraud = (prediction <= 0.5).astype(np.int64)

	is_fraud = np.argmax(prediction, axis=0)
	

	cost_saving = cost_saving(y.values, is_fraud, user_input.values)
	f1Score = f1_score([y.values], [is_fraud]) #labels=np.unique(is_fraud) , zero_division=1

	fraud = " This Transaction is considered Fraudulent! The account would be blocked temporarily. "
	not_fraud = " This transaction is Safe. "



	results_f = {'transact_id': transact_id[0],'cost_saving': cost_saving,'f1Score':f1Score,'prediction': prediction[0],'predict_probability': predict_probability[0][0], 'fraud': fraud}
	results_n = {'transact_id': transact_id[0],'cost_saving': cost_saving,'f1Score':f1Score,'prediction': prediction[0],'predict_probability': predict_probability[0][0], 'fraud':not_fraud}


	if predict_probability[0][0] > 0.5 :
		return render(request,'detector/home.html', results_n)

	else:
		return render(request,'detector/home.html', results_f)





	                     




     


	



	
'''
	shows = []
	if request.method == 'POST':
		Fform =  FraudForm(request.POST, prefix='help')	
		if Fform.is_valid():
			name = Fform.cleaned_data.get('Name')
			v1 = Fform.cleaned_data.get('V1')
			v2 = Fform.cleaned_data.get('V2')
			v3 = Fform.cleaned_data.get('V3')
			v4 = Fform.cleaned_data.get('V4')
			v5 = Fform.cleaned_data.get('V5')
			v6 = Fform.cleaned_data.get('V6')
			v7 = Fform.cleaned_data.get('V7')
			v8 = Fform.cleaned_data.get('V8')
			v9 = Fform.cleaned_data.get('V9')
			v10 = Fform.cleaned_data.get('V10')
			v11 = Fform.cleaned_data.get('V11')
			v12 = Fform.cleaned_data.get('V12')
			v13 = Fform.cleaned_data.get('V13')
			v14 = Fform.cleaned_data.get('V14')
			v15 = Fform.cleaned_data.get('V15')
			v16 = Fform.cleaned_data.get('V16')
			v17 = Fform.cleaned_data.get('V17')
			v18 = Fform.cleaned_data.get('V18')
			v19 = Fform.cleaned_data.get('V19')
			v20 = Fform.cleaned_data.get('V20')
			v21 = Fform.cleaned_data.get('V21')
			v22 = Fform.cleaned_data.get('V22')
			v23 = Fform.cleaned_data.get('V23')
			v24 = Fform.cleaned_data.get('V24')
			v25 = Fform.cleaned_data.get('V25')
			v26 = Fform.cleaned_data.get('V26')
			v27 = Fform.cleaned_data.get('V27')
			v28 = Fform.cleaned_data.get('V28')
			amount = Fform.cleaned_data.get('Amount')
			time = Fform.cleaned_data.get('Time')


			disaster_type = Hform.cleaned_data.get('disaster_type')
			
			Fraud.objects.create(Place=place, Disaster_Type=disaster_type)

		else:
		    Fform = FraudForm()

	results = {'Fform': Fform}
'''