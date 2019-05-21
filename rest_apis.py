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

url_train_model = '/v1/ivr-rec/train'
url_predict_ivr_for_one = '/v1/ivr-rec/predict-one'


@app.route('/trainModel')
def trainModel():
    """
    TODO need to consider to move it to another thread to avoid blocking.
    TODO needs to avoid frequent request.
    """
    with app.app_context():
        result_acc = read_and_train()
        return "{'Final Accuracy': %s, 'error': '%s'}" % (result_acc, "no error")


@app.route('/predictOne')
def predictOne():
    with app.app_context():
        user_json = request.get_json()
        user_arr = get_user_from_json(user_json)

        return "{}"  # TODO need to implement it.

# if __name__ == '__main__':
#     with app.app_context():
#         app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 6006)), debug=True)
