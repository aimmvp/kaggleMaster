
### Package Load
 * pandas : data processing, ex) CSV file I/O //    **http://pandas.pydata.org/**
 * numpy : linear algebra // https://docs.scipy.org/doc/numpy-1.10.0/index.html
 * matplotlib : Graphic Tools / https://matplotlib.org/users/pyplot_tutorial.html
 * re : Regular expression operation // https://docs.python.org/3/library/re.html?highlight=re#module-re

### Data Read
``` python
 train = pd.read_csv('train_1.csv')
```
### data pre processing 
 * 0을 NaN 으로 채운다. 
	```python
 	train = train.fillna(0)
	```
 * 실수(Float)를 정수(Integer)로 바꿔서 메모리 사이즈 줄일수 있음. 정수로 변경해도 괜찮을지 확인 필수
``` python
   for col in train.columns[1:]
   		train[col] = pd.to_numeric(train[col], downcast='integer')
```

### 데이터 처리 하기
* 정규표현식으로 URL 찾기
``` python
	res = re.search('[a-z][a-z].wikipedia.org', page)
```
* iloc 으로 lang에 대해서 index 찾기
```python
lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
```
* 각 언어의 URL 별 평균 구하기
```python
# axis=0 : Row 기준, axis=1 : Column 기준, axis=2 : Depth 기준
# shape[0] 은 Row 기준이기 때문에 해당 key에 대한 Row의 갯수
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
```
* 위에서 구한 평균 값을 기준으로 그래프 그려보기
```python
fig = plt.figure(1, figsize=[10,10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
for key in sums:
	plt.plot(days, sums[key], label = lables[key])
plt.legend()
plt.show()
```
* FFT 를 이용해서 주기적 구조 확인
	- 시계열 데이터 이기때문에 FFT에서의 피크는 주기적인 신호에서 매우 강한 주파수를 보여준다
* 분석 도구들
 - statsmodels 패키지는 시계열 분석을 위한 많은 도구 포함
 - statsmodels : http://www.statsmodels.org/dev/index.html
```python
autocorr = acf(data_diff)	# Autocorrelation function for 1d arrays.
pac = pacf(data_diff)		# Partial autocorrelation estimated
ax1.plot(x[1:],autocorr[1:])	# Lag-N autocorrelation
```

### 예측하기
 * Average number of views for that page (constant value)
 * Linear regression
 * More complicated regression curves
 * ARMA Model(자기회귀이동평균모형, Autoregressive–moving-average model)
 * ARIMA Model(자기회귀 누적 이동평균, Auto-regressive Integrated Moving Average)
  - statsmodels 의 ARIMA 클래스는 매우 느리기 때문에, 병렬처리 하는 것이 좋을 것입니다.