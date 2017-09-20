[ 참고 ]
 - CNN 의 종류 : http://nmhkahn.github.io/Casestudy-CNN

### How far can we go with MNIST??
  - facebook MNIST competition ( https://github.com/hwalsuklee/how-far-can-we-go-with-MNIST/blob/master/README.md )
  
### Kyung Mo Kweon
 - Test error : 0.20%
 - Features : keras, esemble of 3 models (small VGG, small Resnet, very small VGG)
   - keras : Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
   - VGGNet(19 layers)
     - 224 * 224 의 이미지 크기를 입력으로 사용
     - 트레이닝 셋 전체의 RGB 채널 평균 값을 입력 이미지의 각 픽셀마다 substract하여 입력을 zero-centered 되게 한다.
     - 모든 레이어에서 가장 작은 크기의 필터를 사용(3X3, s=1, p=1) 
  - ResNet
     - 매우매우 깊은 레이어를 장하는 네트워크 (152 layers)
     - degradation 문제를 deep residual learning 이라는 학습법으로 해결
 - https://github.com/kkweon/mnist-competition

## Keras Layers
### keras.layers.ZeroPadding2D
```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format=None)
```
```python
net = ZeroPadding2D((3, 3))(input_tensor)
```

 - Zero-padding layer for 2D input (e.g. picture).
 - This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
 - If tuple of 2 ints: interpreted as two different symmetric padding values for height and width


### keras.layers.Conv2D
```python
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
```python
net = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(net)
```
 - 2D convolution layer (e.g. spatial convolution over images).
 - This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
 - When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

### keras.layers.normalization.BatchNormalization
```python
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```
```python
BatchNormalization(name="bn_conv1")(net)
```
Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

### keras.layers.advanced_activations.PReLU
```python
keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```
```python
 PReLU()(net)
```
 - Parametric Rectified Linear Unit.
 - It follows: f(x) = alpha * x for x < 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.
 - ReLU를 일반화 한 것으로, 자신의 파라미터를 학습하여, 30Layers까지 수렴 할 수 있게 해준다.

### keras.layers.pooling.MaxPooling2D
```python
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```
```python
MaxPooling2D((3, 3), strides=(2, 2))(net)
```
 - Max pooling operation for spatial data.
 
### add
```python
add(inputs)
```
```python
x = add([x, sc])
```
 - Functional interface to the Add layer.

### keras.layers.pooling.AveragePooling2D
```python
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```
```python
net = AveragePooling2D((2, 2))(net)
```
 - Average pooling operation for spatial data.

### keras.layers.core.Flatten
```python
keras.layers.core.Flatten()
```
```python
net = Flatten()(net)
```
  - Flattens the input. Does not affect the batch size.


### keras.layers.core.Dense
```python
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
```python
net = Dense(10, activation="softmax", name="softmax")(net)
```
 - Just your regular densely-connected NN layer.
 - Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).


### keras.preprocession.image.ImageDataGenerator
```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    ........... )
```
```python
train_gen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
```
  - Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely.

### plot_model
```python
plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
```
```python
plot_model(model, file_path, show_shapes=True, show_layer_names=False)
```
  - Converts a Keras model to dot format and save to a file.

## argparse
 - recommended command-line parsing module in the Python standard

### ArgumentParser