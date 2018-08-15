# -*- coding: utf-8 -*-
"""

TODO handle the exception

Created on Mon Jun 18 18:54:47 2018

@author: Jim
"""

from model.train_test_util import get_train_test_split2, train
from model.data_clean import save_input_meta_data
from model.mongo_util import read_mongo
import pandas as pd

db_name = 'usrdb'
usr_collection = 'usr'
# mongo_host      = 'ivr-rec-mongo'
mongo_host = 'localhost'
mongo_name = 'mongoadmin'
mongo_pswd = 'secret'


def read_and_train():
    """
    1. read input data
    1.1 sort the columns
    2. drop some unused columns
    3. split X and y
    4. save meta data for X
    5. one-hot-encode X, y
    6. train and save result
    """
    
    # 1. read input data
    # usr_df = read_mongo(db_name, usr_collection, host= mongo_host, username=mongo_name, password=mongo_pswd )
    #  TODO should use the data in DB.
    usr_df = pd.read_csv('./user_train.csv')
    
    # 1.1. sort columns
    usr_df.sort_index(axis=1)
    
    print("fond number of usr records: ", usr_df.shape)
    print("head of usr:\n ", usr_df.head(3))
    
    # 2. drop some unused columns
    usr_df.drop(columns='Spending', inplace=True)
    usr_df.drop(columns='Age', inplace=True)
    
    # 3. split X and y
    usr_y = usr_df.AgeCat
    usr_df.drop(columns='AgeCat', inplace=True)
    
    # 4. save columns meta data for X and y
    save_input_meta_data("X", usr_df)
    save_input_meta_data("y",  usr_y.to_frame("y"))
    print("complete saving input meta data.")
    
    # 5. one-hot-encode X, y

    X_train, X_test, y_train, y_test = get_train_test_split2(usr_df, usr_y)
    
    print("X_train, y_train has a shape of: ", X_train.shape, y_train.shape)
    print("going to train.")
    print('X_train:\n', X_train.head())
    print('y_train:\n', y_train.head())
    # 6. train and save result
    history = train(X_train, y_train)
    print("history: ",history.history)
    print("final accuracy: ", history.history['acc'][-1])
    # TODO save it to mongoDB.
    return history.history['acc'][-1]


if __name__ == "__main__":
    read_and_train()
