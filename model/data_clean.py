# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:01:08 2018

@author: root
"""
import pandas as pd
import json, os, datetime
import numpy as np
import pickle as pickle

from os.path import expanduser

name_working_fodler = expanduser('~')+'/'
name_trian_meta_data = '_meta_data_train.json'


def sort_columns(data):
    '''
    For pandas df
    '''
    return data.sort_index(axis=1, inplace=True)


def save_input_meta_data(x_or_y , d):
    """
    The meta data will be used for one-hot-encoding
    Save meta data for d after load the data from db, but befor training and one-hot encoding.
    the data needs to have its columns sorted already.

    E.G.
        {u'Channel': [0, 1, 2, 5, 3, 4, 6],
         u'Class': [3, 1, 2],
         u'Involoved': [0, 1],
         u'Location': [u'S', u'C', u'Q', nan],
         u'Product': [u'dd', u'C', u'E', u'G6', u'D', u'A', u'B', u'F', u'T'],
         u'Sed': [u'male', u'female'],
         u'SpendingCat': [0, 1, 4, 2, 5, 3],
         u'Title': [u'Mr', u'Mrs', u'Miss', u'Master', u'Rare', u'the Countess']}
    """
    
    usr_meta_data = {}
    
    for c in d.columns:
        a_dict = {c : d[c].unique().tolist()}
        usr_meta_data.update(a_dict)
    
    current_time = str(datetime.datetime.now())
    full_path = name_working_fodler + x_or_y + name_trian_meta_data
    if os.path.exists(full_path):
        os.rename(full_path,
                  name_working_fodler + current_time + x_or_y + name_trian_meta_data)

    with open(full_path, 'w') as f:
        json.dump(usr_meta_data, f)

    #     pickle.dump(usr_meta_data, f)


def load_input_meta_data(x_or_y):
    """
    return the meta data as dict
    see save_input_meta_data for the saved data
    """
    full_path = name_working_fodler + x_or_y + name_trian_meta_data
    with open(full_path, 'r') as f:
        dic = json.load(f)
        return dic


def convert_raw_perdict_usr_data(usr_json):
    """
    usr_json sample:
        {"Involoved":1, "Class":1, "Channel":"0",
         "Product":"C", "Location":"S", "Title":"Mr", "SpendingCat": 1}

    result sample:
        TODO udpate it
        array([u'1.0', u'C', u'1', u'Mr', None, u'S', 1, u'0'], dtype=object)
    """
    metadata = load_input_meta_data()  # FIXME
    usrarr = np.array([])
    usrobj = json.loads(usr_json)
    
    for key in metadata.keys:
        avalue = usrobj.get(key)
        one_hot_value = my_one_hot_encoder(metadata.get(key), avalue)
        usrarr = np.hstack((usrarr, one_hot_value)) 
    
    return usrarr


def my_one_hot_encoder(col_val_enum, data):
    """
    Existing one hot encoders doesn't fit our needs.

    col_val_enum: Encode according to the enum array. enum array E.G. ['Mr', 'Mrs', 'Miss', 'Master']
    data:     Can be n-d array

    retrun a dataframe

    refer to evernote 'python data manipulation' for more
    """
    enum_to_int = dict((t, i) for i, t in enumerate(col_val_enum))
    int_encoded = [enum_to_int[t] for t in data]
    onehot_encoded = list()
    
    for v in int_encoded:
        all_zero_arr = [0 for _ in range(len(col_val_enum)) ]
        all_zero_arr[v] = 1
        onehot_encoded.append(all_zero_arr)
    
    return np.asarray(onehot_encoded)


def my_one_hot_encoder2(col_name, col_val_enum, col_data):
    """
    Existing one hot encoders doesn't fit our needs.
    This method encodes one column.

    col_val_enum: Encode according to the enum array. enum array E.G. ['Mr', 'Mrs', 'Miss', 'Master']
    col_data:         E.G. ['Mr', 'Mrs', 'Miss']  - all data in one column
                  TODO don't support None or NaN

    retrun a dataframe:

        Master  Miss   Mr  Mrs
    0     0.0   0.0  1.0  0.0
    1     0.0   0.0  0.0  1.0
    2     0.0   1.0  0.0  0.0

    refer to evernote 'python data manipulation' for more
    """
    zeros = [0] * len(col_val_enum)
    zero_series = pd.Series(zeros, index=col_val_enum) # TODO append col_name as prefix in index to avoid duplicates
    onehot_encoded = pd.DataFrame([])
    
    for d in col_data:
        tmp = zero_series.copy()
        tmp[d] = 1
        onehot_encoded  = onehot_encoded.append(tmp, ignore_index=True)
        
    return onehot_encoded


def encode_all_col(X):
    return None
