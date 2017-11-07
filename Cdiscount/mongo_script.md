# BSON File Split 하기
* BSON 파일을 Data File 로 사용해야 되는데, 파일 사이즈가 너무 커서 작은 사이즈로 split이 필요해서 찾아본 내용임

* MongoDB 관련 참고 할 만한 사이트
 - https://velopert.com/mongodb-tutorial-list
 - https://docs.mongodb.com
 - tutorial : https://www.tutorialspoint.com/mongodb/mongodb_query_document.htm
 - mongodump : https://docs.mongodb.com/manual/reference/program/mongodump/
 - mongoexport : https://docs.mongodb.com/manual/reference/program/mongoexport/


### Mongo DB 설치
https://docs.mongodb.com/
(for Mac) https://treehouse.github.io/installation-guides/mac/mongo-mac.html

### 전체적인 처리 순서
BSON 파일 다운로드 --> MongoDB 에 Restore --> Data 정보 확인 --> query를 이용한 mongodump 실행 --> 생성된 dump 파일 확인

### Data Restore
1. Kaggle > Data 탭에서 train.bson 파일 다운로드(https://www.kaggle.com/c/cdiscount-image-classification-challenge/data)
2. bson 파일 저장위치로 이동
```
$ cd ~/tensorflow/kaggle_master/Cdiscount
```
3. MongoDB 기동
```
$ mongod
```
4. bson 파일 restore(https://docs.mongodb.com/manual/reference/program/mongorestore/)
```
$ mongorestore --drop -d train_10_5 -c train train.bson
# --drop : target database 에서 collection 을 drop 시킨다.
# --d : database 명
# --c : collection 명
```
5. restore 결과 확인
```
# MongoDB script 실행
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
> db.train.findOne() --> 1개의 데이터 확인
{
	"_id" : 1,
	"imgs" : [
		{
			"picture" : BinData(0,"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgy......
	}
	],
	"category_id" : 1000010653
}
# _id, imgs, category_id 로 구성되어 있음
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

### Validation File 만들기
참고 : https://www.tutorialspoint.com/mongodb/mongodb_query_document.htm
* Training Data 이외에 Validation Data 생성을 위해 원본(train.bson)에서 _id 를 기준으로 200000 미만 170000 초과의 데이터 추출 : 총 12022 건 추출됨
1. id 기준으로 query > mongodump
```
# $lt : less than, $gt : greater than
$ mongodump --db train --collection train --query '{"_id":{$lt:200000, $gt:170000}}'
2017-11-05T22:30:42.001+0900	writing train.train to
2017-11-05T22:30:44.501+0900	done dumping train.train (12022 documents)
```
2. 
https://stackoverflow.com/questions/6996999/how-can-i-use-mongodump-to-dump-out-records-matching-a-specific-date-range

## To-Do : random 대상 추출을 위한 query 작성법 학습 필요
