# coding: utf-8

import pymongo
from pymongo import MongoClient
client = MongoClient('mongodb://ivr-rec-mongo:27017/ivrdb')
client.admin.authenticate('mongoadmin','secret')
db = client.usrdb
usr = db.usr
# get_ipython().magic('save /home/access-mongo')

# to save the csv to mongodb
import pandas as pd
train = pd.read_csv('/home/user_train.csv')

#for doc in train.T.to_dict():
#    value = train.T[doc].to_dict()
#    usr.insert_one(value)
#
# retrive data from mongodb
newdf = pd.DataFrame(list(usr.find()))


