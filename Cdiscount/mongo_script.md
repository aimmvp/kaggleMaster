# BSON File Split 하기
* MongoDB 관련 참고 할 만한 사이트
 - https://velopert.com/mongodb-tutorial-list
 - https://docs.mongodb.com
 - mongodump : https://docs.mongodb.com/manual/reference/program/mongodump/
 - mongoexport : https://docs.mongodb.com/manual/reference/program/mongoexport/


### Mongo DB 설치
https://docs.mongodb.com/
(for Mac) https://treehouse.github.io/installation-guides/mac/mongo-mac.html

### Data Restore
1. Kaggle > Data 탭에서 train.bson 파일 다운로드(https://www.kaggle.com/c/cdiscount-image-classification-challenge/data)
2. 저장위치로 이동
```
$ cd ~/tensorflow/kaggle_master/Cdiscount
```
3. bson 파일 restore
```
$ mongorestore --drop -d train_10_5 -c train train.bson
# --drop : target database 에서 collection 을 drop 시킨다.
# --d : database 명
# --c : collection 명
```
4. restore 결과 확인
```
# MongoDB 기동
$ mongod
> show dbs ==> db 목록 조회
admin           0.000GB
local           0.000GB
train          54.419GB
train_10_5      0.488GB
train_example   0.001GB
> use train
switched to db train
> db.train.count()
7069896
> db.train.find().pretty()
```


### 데이터 잘라서 export하기
1. id가 100000 보다 작은 데이터 dump 파일 생성(*mongoexport 는 JSON, CSV Type으로만 Export 가능*)
```
$ mongodump --db train --collection train --query '{"_id":{$lt:100000}}'
```
2. dump 생성 파일 확인
3. dump 파일 restore
```
$ mongorestore --drop -d train_10_5 -c train train.bson
# -d : database 이름
# -c : collection 이름
```

https://stackoverflow.com/questions/6996999/how-can-i-use-mongodump-to-dump-out-records-matching-a-specific-date-range

## To-Do : random 대상 추출을 위한 query 작성법 학습 필요
