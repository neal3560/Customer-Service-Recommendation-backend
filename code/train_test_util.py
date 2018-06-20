from __future__ import absolute_import
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1

def get_train_test_split(data_X, data_y, test_siz=0.2, random_state=123):
    '''
    input X, y will be processed with one-hot-enoder.
    '''
    print('data_x types:\n', data_X.dtypes)
    data_X_one_hot = pd.get_dummies(data_X)
    spending_cat_dummy = pd.get_dummies(data_X['SpendingCat'])

    print("\n\n* index ***\n\n", data_X_one_hot.index)
    print("\n\n* index2 ***\n\n", spending_cat_dummy.index)
    data_X_one_hot = pd.merge(data_X_one_hot, spending_cat_dummy, on=data_X_one_hot.index, how='outer')
    print("dummies shape: ",data_X_one_hot.shape)
    print("dummies columns: ",data_X_one_hot.columns)

    X_train, X_test, y_train, y_test = train_test_split(\
            data_X_one_hot, data_y, test_size=0.2, random_state=123)

    X_train.drop(columns='SpendingCat', inplace=True)
    X_test.drop(columns='SpendingCat', inplace=True)
    X_test = X_test.add(pd.get_dummies(data_X['SpendingCat']))
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):

    in_dim = X_train.shape[1]
    model = Sequential()

    model.add(Dense(1200, input_dim=in_dim, init='normal', activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(2000, init='normal', activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(10, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' ,metrics = ['accuracy'])
    history = model.fit(X_train, y_train, epochs=3, batch_size=32)
    return history
