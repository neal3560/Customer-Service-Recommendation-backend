#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:34:21 2018

@author: root
"""
from flask import Flask, request
from flask_restful import Resource, Api
from read_and_train import read_and_train
from train_test_util import predict_one, get_user_from_json

import json

app = Flask(__name__)
api = Api(app)

url_train_model = '/v1/ivr-rec/train'
url_predict_ivr_for_one = '/v1/ivr-rec/predict-one'

class TrainModel(Resource):
    '''
    TODO need to consider to move it to another thread to avoid blocking.
    '''
    def get(self):
        result_acc = read_and_train()
        return {'Final Accuracy': result_acc}
    
# class Predict_one(Resource):
#     user_json = request.get_json()
#     user_arr = get_user_from_json(user_json)
    # def put(self):
        
api.add_resource(TrainModel, url_train_model)

if __name__ == '__main__':
    app.run(debug=True)