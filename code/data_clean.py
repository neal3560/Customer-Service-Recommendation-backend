#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:01:08 2018

@author: root
"""
import pandas as pd
import json, os, datetime
import numpy as np

name_trian_meta_data = '/home/' + 'meta_data_train.json'


def sort_columns(data):
    '''
    For pandas df
    '''
    return data.sort_index(axis=1, inplace=True)

def save_input_meta_data(x):
    '''
    the data needs to have its columns sorted already.
    
    E.G.
        {u'Channel': [0, 1, 2, 5, 3, 4, 6],
         u'Class': [3, 1, 2],
         u'Involoved': [0, 1],
         u'Location': [u'S', u'C', u'Q', nan],
         u'Product': [u'XX', u'C', u'E', u'G6', u'D', u'A', u'B', u'F', u'T'],
         u'Sex': [u'male', u'female'],
         u'SpendingCat': [0, 1, 4, 2, 5, 3],
         u'Title': [u'Mr', u'Mrs', u'Miss', u'Master', u'Rare', u'the Countess']}
    '''
    
    usr_meta_data = {}
    
    for c in x.columns:
        a_dict = {c : x[c].unique().tolist()}
        usr_meta_data.update(a_dict)
    
    current_time = str(datetime.now())
    if os.path.exists(name_trian_meta_data):
        os.rename(name_trian_meta_data, current_time + usr_meta_data)
    
    f =  open( name_trian_meta_data, 'w')
    json.dumps(usr_meta_data, f)
    f.close()
    
def load_input_meta_data():
    '''
    return the meta data as dict
    see save_input_meta_data for the saved data
    '''
    f = open(name_trian_meta_data, 'r')
    return dict(str(json.load(name_trian_meta_data, f)))
    
def convert_raw_perdict_usr_data(usr_json):
    '''
    usr_json sample:
        {"Involoved":1, "Class":1, "Channel":"0",
         "Product":"C", "Location":"S", "Title":"Mr", "SpendingCat": 1}
    
    result sample:
        array([u'1.0', u'C', u'1', u'Mr', None, u'S', 1, u'0'], dtype=object)
    '''
    metadata = load_input_meta_data()
    usrarr = np.array([])
    usrobj = json.loads(usr_json)
    
    for key in metadata.keys:
        avalue = usrobj.get(key)
        usrarr = np.hstack((usrarr, avalue)) 
    
    return usrarr

def my_one_hot_encoder(enum_arr, data):
    '''
    Encode according to the enum array. 
    
    refer to evernot 'python data manipulation' for more
    '''
    enum_to_int = dict((t, i) for i, t in enumerate(enum_arr))
    int_encoded = [enum_to_int[t] for t in data]
    onehot_encoded = list()
    
    for v in int_encoded:
        all_zero_arr = [0 for _ in range(len(enum_arr)) ]
        all_zero_arr[v] = 1
        onehot_encoded.append(all_zero_arr)
    
    return onehot_encoded