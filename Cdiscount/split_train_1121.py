import pymongo
from pymongo import MongoClient
from bson.son import SON
import datetime
'''
https://velopert.com/560
db.train.createIndex({category_id:1})
'''

client = MongoClient()
train_db = client.train
train_collection = train_db.train
count_collection = train_db.count_collection
split_collection = train_db.split_collection
# print(train_collection.count())
# 7069896

'''
0. make & export "count_collection"  
 : _id : category_id
    count : count by category_id
    sorting : count desc
    
group_pipe = [
    {'$group': {'_id': '$category_id', 'count': {'$sum': 1}}},
    {'$sort': SON([('count', -1)])},
    {'$out': 'count_collection'}
]
 
count_collection.insert(list(train_collection.aggregate(group_pipe)))

'''

# category_list = count_collection.find({'count': 79640});
# category_list = count_collection.find({'_id': 1000018296});
# 5270
# category_list = count_collection.find({'count': {'$gt':100}});
# 3475

# 1. query category_list
category_list = count_collection.find();

print("######### CATEGORY COUNT : ", category_list.count())

for category in category_list:
    # print(type(category['_id']), ", ", type(category['count']))
    print("START : ", datetime.datetime.now())
    category_id = category['_id']
    cnt = category['count']
    if cnt > 100 :
        cnt = 100
    print("category_id :", category_id, "// cnt: ", cnt)
    split_collection.insert(train_collection.find({'category_id': int(category_id)}).limit(cnt))
    print("After Insert Count : ", split_collection.count())
    print("END : ", datetime.datetime.now())
    print("==============================")

