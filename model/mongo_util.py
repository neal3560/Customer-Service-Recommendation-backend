# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 03:28:14 2018

@author: Jim
"""

import pandas as pd
from pymongo import MongoClient


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s/%s' % (host, port, db)
        conn = MongoClient(mongo_uri)
        conn.admin.authenticate(username, password)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    print('going to access collection: %s, on db:%s' % (collection, db))
    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df


def save_one_user_record(userjson):
    userjson
