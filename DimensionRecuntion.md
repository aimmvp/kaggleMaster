## Demension Reduction
### 0. 이론적인 Dimension Reduction
* 이론적 : 차원수 증가 --> 에러 감소
* 각 클래스의 사전 확률이 같고, 특징의 분포가 정규분포라고 가정할 때, 에러 확률은 마할라노비스 거리로 표현될 수 있음
* 마할라노비스 거리(Mahalanobis distance) : 평균과의 거리가 표준편차의 몇 배인지를 나타내는 값.
  - 어떤 값이 얼마나 일어나기 힘든 값인지, 또는 얼마나 이상한 값인지를 수치화하는 방법
* 고차원(High Dimension)의 문제
  - 에러발생 증가에 따른 계산시간 증가
  - 훈련 데이터 스케일(크기 조정 등 조절)
* 실제 : 차원수 증가 --> 에러가 커지는 상황 발생
  - 훈련 샘플(Training Sample)의 수가 한정적이기 때문
* 신경망 (Neural Network) 학습의 분류
  - 지도학습(supervised learning) : 학습기가 분류 (classification) 하려는 대상이 자갈과 모래라는 것을 미리 알고서 훈련예 (training example) 로서 학습시켜 어떤 대상이 자갈에 속하는지 모래에 속하는지를 분류하는 것
  - 자율학습(unsupervised learning) : 분류하려는 대상에 대한 어떤 정보도 주어지지 않고 학습기로 하여금 그것이 자갈인지 또는 모래인지 또는 그 밖의 어떤 것인지를 분류하는 것

### 1. PCA(Principal Component Analysis, 주성분분석)
  - 변수들의 선형조합(Linear Combination)으로 이루어진 변수를 통해 적은 양의 변수(주성분)으로 전체 변동을 설명하는 방법(여기에서 사용되는 변수는 기존의 변수일수도 있고 변수의 조합일수도 있음)
  - Unsupervised, linear method 에 주로 쓰임
  - 전체변동(variation)을 가장 잘 설명하는 성분을 순차적으로 뽑아냄 --> 데이터의 분산을 최대화하는 방향으로 차원을 축소
  - PCA로 도출된 결과를 바탕으로 이미지를 압축하거나 재건축 가능
  - 참고 : http://bit.ly/2xkcv1I


### 2. LDA(Linear Discriminant Analysis, 선형분류분석)
  - 클래스간 산포(Between-class)와 클래스내 산포(With in-class)를 고려하여 차원을 축소시키는 선형 판별 방법 으로 데이터를 잘 분리하는 방향으로 Projection 시키는 것이 목적
  - Supervised(classification), linear method 에 주로 쓰임

### 3. t-SNE(t-distrinuted Stochastic Neighbour Embedding)
  - 비선형적이고, 확률적인 차수 감소 방법임.
  - PCA, LDA 와 다른 점은 점 사이의 유클리드 거리를 조건부로 변환하는 것을 목표로 하므로써, 확률 분포를 축정하여 하나의 데이터 포인트 간의 유사점을 계산하는 메트릭스에 적용
  - wikipedia : 데이터의 차원 축소에 사용되는 기계학습 알고리즘. 고차원 데이터를 특히 2,3차원 등으로 줄여 가시화하는데에 유용하게 사용. 비슷한 데이터는 선택될 확률이 매우 높지만 다른 데이터끼리는 선택될 확률이 매우 낮도록 설계됨.
  - 