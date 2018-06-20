#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:54:47 2018

@author: Jim
"""

from train_test_util import get_train_test_split
from train_test_util import train
from mongo_util import read_mongo
import pandas as pd

db_name         = 'usrdb'
usr_collection  = 'usr'
mongo_host      = 'ivr-rec-mongo'
mongo_name      = 'mongoadmin'
mongo_pswd      = 'secret'

usr_df = read_mongo(db_name, usr_collection, host= mongo_host, username=mongo_name, password=mongo_pswd )
#usr_df = pd.read_csv('/home/user_train.csv')
print("fond number of usr records: ", usr_df.shape)
print("head of usr:\n ", usr_df.head(3))

usr_df.drop(columns='Spending', inplace=True)
usr_df.drop(columns='Age', inplace=True)
usr_y = usr_df.AgeCat
usr_df.drop(columns='AgeCat', inplace=True)
#usr_df.SpendingCat.astype(categroy)
X_train, X_test, y_train, y_test = get_train_test_split(usr_df, usr_y)

print("X_train, y_train has a shape of: ", X_train.shape, y_train.shape)
print("going to train.")
print('X_train:\n', X_train.head())
print('y_train:\n', y_train.head())
history = train(X_train, y_train)
print("history: ",history.history)
