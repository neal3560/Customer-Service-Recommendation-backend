# from __future__ import absolute_import
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from data_clean import sort_columns, my_one_hot_encoder2, load_input_meta_data
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1
import datetime, os, json

def get_train_test_split(data_X, data_y, test_siz=0.2, random_state=123):
    '''
    input X, y will be processed with one-hot-enoder.
    TODO: move the get dummies in the last step.
    '''
    
    sort_columns(data_X)
    
    print('data_x types:\n', data_X.dtypes)
    data_X_one_hot = pd.get_dummies(data_X)
    dummy = pd.get_dummies(data_X['SpendingCat'])

    data_X_one_hot = pd.concat([data_X_one_hot, dummy], axis=1)
    print("dummies shape: ",data_X_one_hot.shape)
    print("dummies columns: ",data_X_one_hot.columns)

    X_train, X_test, y_train, y_test = train_test_split(
                                            data_X_one_hot, data_y,
                                            test_size=0.2, random_state=123)

    X_train.drop(columns='SpendingCat', inplace=True)
    X_test.drop(columns='SpendingCat', inplace=True)
    X_test = X_test.add(pd.get_dummies(data_X['SpendingCat']))
    # TODO dummies encoder is ok for now until we need to explain the prediction
    y_train = pd.get_dummies(y_train) 
    y_test = pd.get_dummies(y_test)
    return X_train, X_test, y_train, y_test

def get_train_test_split2(data_X, data_y, test_siz=0.2, random_state=123):
    '''
    We must use our own version of one-hot-encoder to ensure both training and
    prediction data are treated the same way.
    
    input X, y will be processed with one-hot-enoder.
    TODO: move the get dummies in the last step.
    '''
    
    sort_columns(data_X)
    
    print('data_x types:\n', data_X.dtypes)
    
    # TODO move this part to a function
    X_meta = load_input_meta_data('X')
    y_meta = load_input_meta_data('y')
    data_X_one_hot = data_X
    
    col_names = ['SpendingCat', 'Sex', 'Product', 'Title']
    col_names = np.sort(col_names)
    for n in col_names:
        dummy = my_one_hot_encoder2(n, X_meta[n], data_X[n])
        data_X_one_hot = pd.concat([data_X_one_hot, dummy], axis=1)
        data_X_one_hot.drop(n, inplace = True)
    
    print("dummies shape: ", data_X_one_hot.shape)
    print("dummies columns: ", data_X_one_hot.columns)

    X_train, X_test, y_train, y_test = train_test_split(\
            data_X_one_hot, data_y, test_size=0.2, random_state=123)

    y_train = (y_train)
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
    _save_model(model)
    
    return history

model_name = '/home/trained_model.h5'
def _save_model(model):
    
    current_time = str(datetime.now())
    
    if os.path.exists(model_name):
        os.rename(model_name, current_time + model_name)
    model.save(model_name)
    
model_instance = None

def _get_trained_model():
    if model_instance is None:
       model_instance = load_model(model_name)
    return model_instance
    
def predict_one(user_behavior):
    '''
    user_behavior is the numpy array.
    '''
    proba = _get_trained_model.predict_proba(user_behavior)
    print(proba)
    # get the top 3
    return np.argpartition(proba[0], -3)[-3:]

def predict_many(user_behavior):
    '''
    Not implemented
    '''
    return None

def get_user_from_json(user_json):
    '''
    Sample json contains following:
    Involoved,Class,Sex,Age,Channel,Spending,Product,Location,Title,
    SpendingCat,AgeCat
    
    TODO:
    The order of columns should be:
        Channel               int64
        Class                 int64
        Involoved             int64
        Location_C            uint8
        Location_Q            uint8
        Location_S            uint8
        Product_A             uint8
        Product_B             uint8
        Product_C             uint8
        Product_D             uint8
        Product_E             uint8
        Product_F             uint8
        Product_G6            uint8
        Product_T             uint8
        Product_XX            uint8
        Sex_female            uint8
        Sex_male              uint8
        SpendingCat_0         uint8
        SpendingCat_1         uint8
        SpendingCat_2         uint8
        SpendingCat_3         uint8
        SpendingCat_4         uint8
        SpendingCat_5         uint8
        Title_Master          uint8
        Title_Miss            uint8
        Title_Mr              uint8
        Title_Mrs             uint8
        Title_Rare            uint8
        Title_the Countess    uint8
    '''
    user_obj = json.loads(user_json)
    