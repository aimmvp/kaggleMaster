import pymongo
from pymongo import MongoClient
import bson
from bson.son import SON
import datetime
import io
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import cv2
import numpy as np
'''
https://velopert.com/560
db.train.createIndex({category_id:1})
'''

# 1. Define DB, Collection
# original DB : train
# original Collection : train --> train_collection
# count collection : count by category_id --> count_collection
# made collection --> split_collection

client = MongoClient()
train_db = client.train
train_collection = train_db.train
count_collection = train_db.count_collection
split_collection = train_db.split_collection
# create a index by category_id
# train_collection.create_index("category_id")
print(" Count of train_collection : ", train_collection.count())
# for index in train_collection.list_indexes():
#     print(index)
#7069896


'''
# 2. Make count_collection
# order by count desc
# output --> count_collection
count_pipe = [
    {'$group': {'_id': '$category_id', 'count': {'$sum': 1}}},
    {'$sort': SON([('count', -1)])},
    {'$out': 'count_collection'}
]
print("Count START : ", datetime.datetime.now())
print("Before Delete --> count_collection : ", count_collection.count())
# count_collection.delete_many({})
print("After Delete --> count_collection : ", count_collection.count())
# count_collection.insert(list(train_collection.aggregate(count_pipe)))
print("After Insert --> count_collection : ", count_collection.count())
print("Count END : ", datetime.datetime.now())
# 5270
'''
# 3.
