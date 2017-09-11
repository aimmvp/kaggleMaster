참고 URL : https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration

이 노트북은 이 대회의 학습데이터에 대한 몇가지 시각화를 다룬다. 우리는 간단한 회귀모델에 대한 해석을 확대 할지에 대한 몇가지 견해(insight)를 얻을 수 있고, 좋은 예측을 하는데에서 많은 잠재적 어려움이 있음을 확인 할 수 있다.

먼저 몇가지 패키지를 load 한다.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
```

* 참고한 패키지 
 - pandas : data processing, ex) CSV file I/O // http://pandas.pydata.org/
 - numpy : linear algebra / https://docs.scipy.org/doc/numpy-1.10.0/index.html
 - matplotlib : Graphic Tools /  https://matplotlib.org/users/pyplot_tutorial.html
 - re : Regular expression operation /  https://docs.python.org/3/library/re.html?highlight=re#module-re
 - Matplotlib tutorial : http://www.labri.fr/perso/nrougier/teaching/matplotlib/
---

학습데이터를 read 한다.
``` python
train = pd.read_csv('train_1.csv')
```

데이터셋은 0과 missing value 를 구분하지 않기 때문에 0 을 NaN 으로 채운다. 상위 데이터를 보여준다.(default 5)

``` python
train = train.fillna(0) 
train.head()
```

메모리 절약을 위해 모든 값을 정수(Integer)로 바꿀것이다. Pandas 에서, 파일을 읽으면서 자동으로 정수를 NaN 으로 만들 수는 없기때문에, 나중에 할 것이다. 이것은 메모리에서의 사이즈를 600Mbyte 에서 300Mbyte 로 줄여준다. 실수에 대한 예측을 원할지도 모르지만, 보여지는 것은 어쨓든 정수이기 때문에 어떤 정보도 놓치지 않는다.  

``` python
# 몇가지 오류가 발생하여 일단 주석처리한다. 
# for col in train.columns[1:]:
#	train[col] = pd.to_numeric(train[col], downcast='integer')
# train.head()
```

``` python
train.info()
```

### 페이지의 언어에 의해서 Traffic 이 영향을 받을까?

이 것을 보면서 관심을 가질 수 있는 것 중에 한가지는 Wikipedia 에서 사용되는 다른 언어가 어떻게 dataset 에 영향을 주는지이다. Wikipedia URL 에서 언어코드를 찾아내기 위해 간단한 정규표현식을 사용할 것이다. 위키피디아 URL 이 아닌 많은 URLdms 정규표현식 검색이 안되는 것도 많다. 위키피디아 페이지 중에서 페이지의 언어를 결정 할 수 없는 경우에는 'na' 로 넣을 것이다. 이들 중 많은 부분은 실제로 언어를 가지고 있지 않은 것처럼 여길 것이다. 

실제 wikipedia 의 URL : https://ko.wikipedia.org/wiki/sum

``` python
# 정규표현식을 이용하여 URL 의 언어 추출
def get_language(page):
	res = re.search('[a-z][a-z].wikipedia.org', page)
	if res:
		return res[0][0:2]	# 첫 2글자를 잘라냄 ex) en, de
	return 'na'
	
# train map 의 가장 끝에 lang 이라는 컬럼을 만들고 return 값을 추가
train['lang'] = train.Page.map(get_language)

from collections import Counter
print(Counter(train.lang))
# Counter({'en': 24108, 'ja': 20431, 'de': 18547, 'na': 17855, 'fr': 17802, 'zh': 17229, 'ru': 15022, 'es': 14069}
```

7가지 이상의 언어와 미디어 페이지가 있다. 사용된 언어는 English, Japanese, German, French, Chinese, Russian, and Spanish 가 있다. 이렇게 하면 처리할 수 있는 4가지 다른 writing system(Latin, Cyrillic, Chinese, and Japanese) 이 있기 때문에 URL 분석이 어려워질 것이다. 여기에서 초반의 다른 Type 을 위한 dataframe 을 만들것이고, 모든 Views(traffic??)의 합을 계산 할 수 있다. 몇가지의 다른 source 에서 데이터를 추출 하기 때문에 합은 일부 View 에 대해서 두배로 계산할 가능성이 높다.

```python
lang_sets = {}
# iloc 에서 앞에 : 은 모든 lang 에 대해서 index  찾기
# 0:-1 은 제일 끝에 한개 빼고
# 차이는......
lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train[train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train[train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train[train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train[train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train[train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train[train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train[train.lang=='es'].iloc[:,0:-1]

sums = {}
# key 로 language 를 뽑아내고,
# sum(axis=0)에서 axis=0 은 Row 를 기준으로 라는 뜻이며, language 를 나타낸다.
# axis=0 : Row 기준, axis=1 : Column 기준, axis=2 : Depth 기준
# shape[0] 은 Row 기준이기 때문에 해당 key에 대한 Row의 갯수
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
```
* 결국은 평균???? 

시간의 흐름에 따라 총 조회 수는 어떻게 변할까? 하나의 plot에 다른 모든 lang_sets 을 그린다.

### matplotlib.pyplot 을 이용해서 그래프 그리기

```python
# English 의 갯수를 기준으로 Days array 를 만듬 : 0 ~ 549
days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[10,10])	# 1: num of figure, figsize : width, height in inches
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
# label  매핑
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'
       }

# key--> language 에 따른 sum을 그래프로 출력
for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()
```

Wikipedia는 English기반 사이트이기 때문에 영어는 페이지당 훨씬 많은 조회수를 보인다. 기대했던 것보다 훨씬 많은 구조가 있다. 영어와 러시아어 그래프는 400일(2016년 8월경) 에서 매우 큰 스파이크를 보여준다. 2016년에 나중에 영어 데이터에서 몇개의 스파이크가 추가된다. 이것은 미국에서의 8월 하계올림픽과 선거의 영향으로 추측된다.
200일 부근에 영어에서 또 다른 특이한 양상이 나타난다.
스페인어 데이터 또한 매우 흥미롭다. 거기에는 1주정도의 빠른 주기와 매 6개월 부근에서 확실히 깁어지는 것과 같은 확실한 주기적 구조가 있다.  

### 주기적 구조와 FFTs
여기에 몇가지 주기적인 구조가 있는 것 같기 때문에 이것 각각을 따로 그려서 눈금보다 더 잘 보일 것이다. 각각의 그래프와 함께, FFT의 크기도 살펴 볼 것이다. FFT에서의 피크는 주기적인 신호에서 매우 강한 주파수를 보여준다. 
```python
def plot_with_fft(key) :
	# 1. the 1st Figure
    fig = plt.figure(1, figsize=[15,5])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.title(labels[key])
    plt.plot(days, sums[key], label = labels[key])

	# 2. Compute FFTs
    fig = plt.figure(2, figsize=[15,5])
    fft_complex = fft(sums[key])
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2+1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]

	# 3. Set Figure
    plt.ylabel('FFT Magnitude')
    plt.xlabel(r"Frequency [days]$^{-1}$")
    plt.title('Fourier Transform')
    plt.plot(fft_xvals[1:], fft_mag[1:], label = labels[key])
    # Draw lines at 1, 1/2, and 1/3 week periods
	# axvline : Add a vertical line across the axes
	# alpha : float(0.0 transparent through 1.0 opaque)
    plt.axvline(x=1./7, color='red', alpha=0.3)
    plt.axvline(x=2./7, color='red', alpha=0.3)
    plt.axvline(x=3./7, color='red', alpha=0.3)

    plt.show()

for key in sums:
    plot_with_fft(key)

```

이 것을 통해서 우리는 스페인어 데이터가 가장 강한 주기적인 모형을 가지고 있지만 다른 언어의 대부분도 몇가지 주기성을 보여준다. 몇가지 이유로 러시아어와 미디어 데이터(?)는 많이 보이지 않는 것 같다. 나는 1, 1/2, 1/3 주의 기간이 나타나도록 빨간 선을 그렸다. 우리는 주기적인 모형이 1, 1/2 주에서 주로 보인다. 이것은 검색 습관은 주말과 비교해서 주중이 다를 것이기 때문에 정수 n에 대해서 n/(1 week)의 주기가 FFT에서 최고점으로 이어지는 것이 놀라운 일은 아니다. 우리는 페이지 뷰가 모두다 원활하지는 않다는 것을 알았다.  몇가지 주기적인 변화가 일상적으로 있지만, 갑자기 발생할 수 있는 큰 영향도 있다. 모델은 그 날 세상에서 일어 날 것에 대한 더 많은 정보가 제공 되지 않는 한 갑작스런 급증은 예측 할 수 없을 것이다. 

### Individual Entry Data(각각의 입력 데이터)
몇가지 개별 입력데이터를 그릴 것이다. 관찰을 위해 몇가지 입력을 뽑았지만, 그것들에 대해서 특별한 것은 필요없다.

```python
def plot_entry(key,idx):
	# data : lang_sets에서 언어(key)의 set을 추출 --> idx에 해당하는 URL 선택 --> 1: 은 URL을 제외한 전체 데이터
    data = lang_sets[key].iloc[idx,1:]	
    fig = plt.figure(1,figsize=(10,5))
    plt.plot(days,data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(train.iloc[lang_sets[key].index[idx],0])
    
    plt.show()

idx = [1, 5, 10, 50, 100, 250,500, 750,1000,1500,2000,3000,4000,5000]
for i in idx:
    plot_entry('en',i)
```

개별 페이지에서 데이터도 부드럽지 않다. 갑자기 엄청난 스파이크, 뷰의 평균수에서 큰 이동 등이 있다. 또한 위키피디아 관점에서 뚜렷한 이벤트의 영향을 확실하게 볼 수 있다.
2016년 북한 핵실험이 발생하고, 위키피디아 페이지가 빨리 구축되고, 짧은 시간에 엄청난 수의 조회(View)가 있었다. 조회수는 1~2주후에 주로 사라졌다.
Hunky Dory는 2016년 초반에 엄청난 수의 조회를 받았고, David Bowie의 죽음에 해당한다.
2016 올림픽에서 사격시합에 대한 페이지는 적은 수의 조회를 가졌고,  올림픽을 중심으로 갑작스런 조회를 보였다. 
Fiji Water의 데이터에서의 두개의 거대한 급증, "Internet of Things" 와 "Credit score"의 트래픽이 갑자기 장기간 증가와 같은 몇가지 특이한 것이 있다. 그 즈음 Fiji water에 대한 몇가지 뉴스가 있었을 수도 있다.
다른 편에서, 검색엔진 동작에서의 변경이 있었거나, 매우 눈에 잘 띄는 곳에 링크가 생겼기 때문 일 수있다.
몇가지 스페인어 입력을 보자.

```python
idx = [1, 5, 10, 50, 100, 250,500, 750,1001,1500,2000,3000,4000,5000]
for i in idx:
    plot_entry('es',i)
```

스페인어가 영어 데이터보다 훨씬 극단적인 짧은 스파이크가 나온다. 만약 이것들 중 일부는 평균으로 돌아가기 1~2일 전이라면, 데이터에서 뭔가가 잘못되었음의 신호 일 것이다. 우리가 거의 예측이 불가능한 매우 짧은 스파이크를 처리하기 위해, 중간값 필터가 사용 되어 그 필터를 제거 할 수 있다.
여기에서 매우 흥미로운 것이 있다. 매우 강한 주기적 구조는 특정 페이지에서만 나타난다. 가장 강한 주기적 구조를 나타내는 구조를 보여주는 플롯은 실제로 공통점이 몇가지 있다. - 이것들 모두는 건강 주제와 관련된 것으로 보인다. 주간 구조는 사람들이 의사를 보고 위키피디아를 참고하는 것과 연관되어 있다면 의미가 있다. 더 긴 구조(~6month) 는 특히, 어떤 브라우저 인구 통계 정보를 가지지 않으면 설명하기 어렵다. 

프랑스 도표는 더 많은 것을 보여준다. 위키피디아 조회는 뉴스에 어떤것이 있고 없고에 따라 많이 달라진다. Leicester FC가 프리미어 리그에서 우승하고, 챔피언십에 대한 많은 조회를 받았다. 올림픽은 그 페이지에 대한 트래픽에서 매우 큰 증감을 야기한다. 크리스마스는 대림절동안 조회수가 꾸준히 증가하는 것과 같은 흥미로운 구조를 보여준다. 

### How does the aggregated data compare to the most popular pages?(집계된 데이터는 가장 유명한 페이지와 어떻게 비교되는가?)
집계된 데이터의 잠재적인 몇가지 문제를 언급했었다. 이 데이터세트에서 언어의 메인페이지가 될 가장 유명한 페이지를 볼 것이다. 
 --> 언어별로 가장 인기 있는 페이지 조회
```python
# For each language get highest few pages
npages = 5
top_pages = {}
for key in lang_sets:
    print("key ::::::", key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)	# axis=1 : 해당 로우의 값들을 더함
    sum_set = sum_set.sort_values('total',ascending=False)	# sort_values : Sort by the values along either axis
    print(sum_set.head(10))
#	                                                    Page         total
#	38573   Main_Page_en.wikipedia.org_all-access_all-agents  1.206618e+10
#	9774       Main_Page_en.wikipedia.org_desktop_all-agents  8.774497e+09
    top_pages[key] = sum_set.index[0]		# Descending sort ==> 0번째 : 제일 많은 URL 의 index
	print(top_pages)		# {'en': 38753}	
    print('\n\n')
```

위에서 만든 인기페이지 그래프 그리기
```python
for key in top_pages:
    fig = plt.figure(1,figsize=(10,5))
    cols = train.columns
    cols = cols[1:-1]
    data = train.loc[top_pages[key],cols]
    plt.plot(days,data)
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.title(train.loc[top_pages[key],'Page'])
    plt.show()
```

이것들을 집계된 데이터와 비교해보면 거의 비슷함을 알 수 있다. 올림픽이 위키피디아 같은 사이트에 엄청나게 큰 영향을 줄 수 있다는 것이 매우 놀랍다. 나는 일본어, 스페인어, 미디어 데이터가 가장 다르다고 말할것이다. 미디어 페이지에서, 대부분의 사람이 메인페이지나 검색을 사용하는 것 대신, 링크를 통해 페이지에 접속 할것으로 예상된다. 일부 언어는 메인페이지와 집계데이터에 많은 차이를 보인다는 사실은 데이터셋이 위키피디아로의 모든 트래픽을 대표하는 것은 않을 수 있음을 나타낸다.

### More Analysis Tools
<b>statsmodels</b> 패키지는 시계열 분석은 꽤 많은 도구를 포함한다. 여기서, 각 언어별로 가장 많이 본 페이지의 자기상관과 부분자기상관을 보여준다. 이 두가지 모두 신호와 자신의 지연된 버전의 상호관계를 보여준다. 각 지연에서, 부분자기상관은 더 짧은 지연에서 상관관계를 제거 후 그 상관관계를 보여준다.

statsmodels : http://www.statsmodels.org/dev/index.html
```python
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

for key in top_pages:
    fig = plt.figure(1,figsize=[10,5])
    ax1 = fig.add_subplot(121) 	# add_subplot(121) ==> add_subplot(1,2,1) ==> 1*2 그리드에서 1번째 subplot
    ax2 = fig.add_subplot(122)
    cols = train.columns[1:-1]	# URL 제외, 제일 끝에 있는 total 제외
    data = np.array(train.loc[top_pages[key],cols])
    data_diff = [data[i] - data[i-1] for i in range(1,len(data))]
    autocorr = acf(data_diff)	# Autocorrelation function for 1d arrays.
    pac = pacf(data_diff)		# Partial autocorrelation estimated

    x = [x for x in range(len(pac))]
    ax1.plot(x[1:],autocorr[1:])	# Lag-N autocorrelation

    ax2.plot(x[1:],pac[1:])
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title(train.loc[top_pages[key],'Page'])

    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    plt.show()
```

대부분의 경우 주간효과로 7일마다 강한 상관관계와 회귀상관관계를 보인다.부분자기상관의 경우, 첫주가 가장 강하고 그 다음에 낮아지기 시작한다.

### Making Predictions
지금까지 주어진 기준선은 모든 것에 대해 0뷰를 예측하는 것이다. 우리가 해볼수있는 몇가지 쉬운 벤치마크가 있다.
 * Average number of views for that page (constant value)
 * Linear regression
 * More complicated regression curves

그러나 관련 주제의 데이터가 자기산관관계가 있고, 뉴스에 있는 주제가 많은 트래픽은 유발시키는 것으로 보인다. 따라서 이 것은 일을 해결하는 몇가지 방법을 가리킨다. 불행히, 다른 언어로 관련 주제를 확인하기 위한 모델을 훈련시키는 것은 꽤 어려울 것이다. 그러나 페이지 이름보다 데이터만 사용하여 비슷한 주제끼리 분류하면 약간 도움이 될 수 있다. 우리는 어쩌면 예상할 수 없는 스파이크의 일부를 부드럽게 제거 할 수 있고, 조회수가 낮은 페이지에서 통계적인 변동의 영향을 줄이기 위해 조회수가 높은 페이지를 사용 할 수 있을 것이다. 만약 우리가 더 복잡한 방법의 시도롤 원한다면 RNN과 같은 것이 여기에서는 도움이 될지 궁금하다.
예를 들면, ARIMA 모델에서 작은 페이지 세트를 찾는다.
* ARIMA(자기회귀 누적 이동평균, Auto-regressive Integrated Moving Average) 모형 : 시계열 분석 기법의 한 종류로, 과거의 관측값과 오차를 사용해서 현재의 시계열 값을 설명하는 ARMA 모델을 일반화 한 것입니다. 이는 ARMA 모델이 안정적 시계열(Stationary Series)에만 적용 가능한 것에 비해, 분석 대상이 약간은 비안정적  시계열(Non Stationary Series)의 특징을 보여도 적용이 가능하다는 의미입니다. (참고 : http://www.dodomira.com/2016/04/21/arima_in_r/)

### ARIMA Models
Statsmodels 는 시계열예측에 사용 될 수 있는 ARMA 와 ARIMA 모델같은 것도 포함하고 있다. 이 데이터는 반드시 움직임이 없는 것은 아미고 주기적인 영향이 강하기 때문에 반드시 잘 작동하는 것은 아닙니다. 높은 조회수의 페이지의 같은 집합에 대한 ARIMA 예측을 살펴볼것이다.

```python
from statsmodels.tsa.arima_model import ARIMA
import warnings

cols = train.columns[1:-1]	# URL 제외, total 제외 한 날짜만 추출
for key in top_pages:
    data = np.array(train.loc[top_pages[key],cols],'f')		# 데이터를 f : float 으로 바꿈
    result = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            arima = ARIMA(data,[2,1,4])
			# Fits ARIMA(p,d,q) model by exact maximum likelihood via Kalman filter
            result = arima.fit(disp=False)
        except:
            try:
                arima = ARIMA(data,[2,1,2])
                result = arima.fit(disp=False)
            except:
                print(train.loc[top_pages[key],'Page'])
                print('\tARIMA failed')
    #print(result.params)
    pred = result.predict(2,599,typ='levels')
    x = [i for i in range(600)]
    i=0

    plt.plot(x[2:len(data)],data[2:] ,label='Data')
    plt.plot(x[2:],pred,label='ARIMA Model')
    plt.title(train.loc[top_pages[key],'Page'])
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()
```

몇가지 경우에서 ARIMA 모델이 신호의 주간구조를 잘 예측 할 수 있었다. 다른 경우에는 선형 적합으로 나타내는 것처럼 보인다. 이것은 잠재적으로 매우 유용하다.
그러나 전체 데이터셋에 맹목적으로 ARIMA 모델을 적용한다면, 그 결과는 단순히 기본적인 중앙값 모델을 적용하는 것보다 좋지 않을 것입니다. 여전히 흥미로운 속성을 가지고 있는 것으로 보이므로, 더 좋은 결과를 얻기 위해 다른 모델과 결합할 수 있을 것입니다. 또는, ARIMA 가 다른 모델보다 더 잘 작동할것으로 기대되는 데이터의 하위집합을 찾을 수 있다. 
불행하게도, statsmodels 의 ARIMA 클래스는 매우 느리기 때문에, 병렬처리 하는 것이 좋을 것입니다.