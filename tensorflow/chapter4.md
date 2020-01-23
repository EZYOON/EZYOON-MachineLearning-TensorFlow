# chapter 4

## ch4.2

### 간단한 분류 모델 구현하기

#### 털과 날개가 있는냐를 기준으로 포유류와 조류를 구분하는 신경망 모델을 만들어 보자

#### Numpy 라이브러리는 유명한 수치해석동 파이썬 라이브러리 이다. 행렬 조작과 연산에 필수라고 할수 있다.


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
```

    WARNING:tensorflow:From c:\users\jiyun lee\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]]) 
#[털, 날개] 있으면 1 없으면 0
```


```python
y_data = np.array([[1,0,0], #기타
                  [0,1,0], #포유류
                  [0,0,1], #조류
                  [1,0,0],
                  [1,0,0],
                  [0,0,1]])
```

#### [털, 날개] -> [기타, 포유류, 조류]
#### [0,0] -> [1,0,0] 기타
#### [1,0] -> [0,1,0] 포유류
#### [1,1] -> [0,0,1] 조류
#### [0,0] -> [1,0,0] 기타
#### [0,0] -> [1,0,0] 기타
#### [0,1] -> [0,0,1] 조류


```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
```


```python
W = tf.Variable(tf.random_uniform([2,3],-1.,1.))
b = tf.Variable(tf.zeros([3]))
```

#### +) 가중치 변수 W는 [입력층(특징 수),출력층(레이블 수)]의 구성인 [2,3]으로 설정하고 편향 변수 b는 레이블 수인 3개의 요소를 가진 변수로 설정합니다.


```python
L = tf.add(tf.matmul(X,W),b)
L = tf.nn.relu(L)
```


```python
model = tf.nn.softmax(L)
```

#### +) 신경망을 통해 나온 출력값을 softmax 함수를 이용하여 사용하기 쉽게 다듬어준다. 
#### softmax 는 배열 내의 결과값는들 전헤 합이 1이 되도록 만들어 준다 ex) [8.14, 2.27, -6.52] -> [0.53, 0.24, 0.23]


```python
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model),axis=1))
```

#### +) reduce_xxxx 함수는 텐서의 차원을 줄여준다. xxxx 부분이 구체적인 차원 축소 방법을 뜻하고 axis 매개변수로 축소할 차원을 정한다.


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)
```


```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```


```python
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data,Y: y_data})
    if (step + 1) % 10 ==0:
        print(step + 1,sess.run(cost, feed_dict={X: x_data, Y: y_data}))
```

    10 1.0348145
    20 1.0286428
    30 1.0226768
    40 1.0167947
    50 1.011109
    60 1.0054673
    70 1.0000067
    80 0.9946416
    90 0.98937345
    100 0.9842555
    

#### argmax 는 행에서 가장 큰 원소가 위치하는 열의 인덱스를 반환 


```python
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실측값:', sess.run(target, feed_dict={Y: y_data}))
```

    예측값: [2 1 2 2 2 2]
    실측값: [0 1 2 0 0 2]
    


```python
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    정확도: 50.00
    

#### +) tf.cast함수로 0과 1로 바꾸어 평군을 내면 간단히 정확도를 구할 수 있다.

#### 학습 횟수를 아무리 늘려도 정확도가 크게 높아지지 않을 것. 그 이유는 신경망이 한 층밖에 안되기 때문

## ch4.3

### 심층 신경망 구현하기

#### 신경망의 층을 둘 이상으로 구성한 심층 신경망, 즉 딥러닝을 구현해보자 다중신경망을 만드는 것은 신경망 모델에 가중치와 편향을 추가하면 된다.

#### 입력층과 출력충은 각각 특징과 분류 개수로 맞추고, 중간 연결 분분은 맞닿은 층의 뉴선수와 같도록 맞춘다. 중간의 견결 부분을 은닉층이라고 하고 은닉층의 뉴런 수는 하이퍼파라미터이니 실험을 통해 가장 적절한 수를 정하면 된다.


```python
W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.)) #[특징, 은닉층의 뉴런 수]
W2 = tf.Variable(tf.random_uniform([10,3],-1.,1.)) #[은닉층의 뉴런 수, 분류 수]
```


```python
b1 = tf.Variable(tf.zeros([10])) #은닉층의 뉴런수
b2 = tf.Variable(tf.zeros([3])) #분류 수
```


```python
L1 = tf.add(tf.matmul(X,W1),b1)
L1 = tf.nn.relu(L1)
```


```python
model = tf.add(tf.matmul(L1,W2),b2)
```


```python
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y,logits=model))
```


```python
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
```


```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data,Y: y_data})
    if (step + 1) % 10 ==0:
        print(step + 1,sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실측값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    10 1.4652691
    20 1.0922796
    30 0.87732786
    40 0.7371385
    50 0.61609536
    60 0.50270486
    70 0.4050025
    80 0.31840667
    90 0.2517248
    100 0.20361434
    예측값: [0 1 2 0 0 2]
    실측값: [0 1 2 0 0 2]
    정확도: 100.00
    
