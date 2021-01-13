from django.apps import AppConfig
from django.conf import settings

import gzip
import dill
import os
import json
import joblib

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DetectorConfig(AppConfig):
    name = 'detector'

    # de serializing the model
    model_path = os.path.join(settings.MODEL_TOOLS, 'fraud_model.bin')
    #bst = xgb.Booster({'nthread': 4})  # init model
    model = XGBClassifier(random_state=1)
    #fraud_model = model.load_model(model_path)
    model.load_model(model_path)


    # de-serializing the data preparation pipeline
    model_path = os.path.join(settings.MODEL_TOOLS, 'data_prep_pipe.gz.dill')
    dill._dill._reverse_typemap['ClassType'] = type # Had to add this to stop a strange error from modern dill packages

    with gzip.open(model_path, 'rb') as f:
        data_prep_pipe = dill.load(f) 

    # de-serializing the data preparation pipeline
    model_path = os.path.join(settings.MODEL_TOOLS, 'data.gz.dill')
    dill._dill._reverse_typemap['ClassType'] = type # Had to add this to stop a strange error from modern dill packages

    with gzip.open(model_path, 'rb') as f:
        data = dill.load(f) 

    # de-serializing the calibration model
    model_path = os.path.join(settings.MODEL_TOOLS, 'calibration.gz.dill')
    dill._dill._reverse_typemap['ClassType'] = type # Had to add this to stop a strange error from modern dill packages

    with gzip.open(model_path, 'rb') as f:
        calibration = dill.load(f) 



    
            
