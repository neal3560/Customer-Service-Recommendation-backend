# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:34:21 2018

@author: root
"""
import os

from flask import Flask, request
from flask_restful import Resource, Api

from model.train_test_util import get_user_from_json
from read_and_train import read_and_train

app = Flask(__name__)
api = Api(app)

url_train_model = '/v1/ivr-rec/train'
url_predict_ivr_for_one = '/v1/ivr-rec/predict-one'

class TrainModel(Resource):
    """
    TODO need to consider to move it to another thread to avoid blocking.
    """

    def get(self):
        result_acc = read_and_train()
        return {'Final Accuracy': result_acc}


class PredictOne(Resource):
    user_json = request.get_json()
    user_arr = get_user_from_json(user_json)


    def get(self):
        return {}  # TODO need to implement it.


api.add_resource(TrainModel, url_train_model)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 6006)), debug=True)