[ 참고 ]
 - CNN 의 종류 : http://nmhkahn.github.io/Casestudy-CNN

### How far can we go with MNIST??
  - facebook MNIST competition ( https://github.com/hwalsuklee/how-far-can-we-go-with-MNIST/blob/master/README.md )
  
### Kyung Mo Kweon
 - Test error : 0.20%
 - Features : keras, esemble of 3 models (small VGG, small Resnet, very small VGG)
   - VGGNet(19 layers)
     - 224 * 224 의 이미지 크기를 입력으로 사용
     - 트레이닝 셋 전체의 RGB 채널 평균 값을 입력 이미지의 각 픽셀마다 substract하여 입력을 zero-centered 되게 한다.
     - 모든 레이어에서 가장 작은 크기의 필터를 사용(3X3, s=1, p=1) 
  - ResNet
     - 매우매우 깊은 레이어를 장하는 네트워크 (152 layers)
     - degradation 문제를 deep residual learning 이라는 학습법으로 해결
 - https://github.com/kkweon/mnist-competition