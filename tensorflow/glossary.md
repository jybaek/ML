# 용어 정리
알파벳 순서로 정리합니다. 

## A
## B
## C
### CNN (Convolutional Neural Network, 합성곱 신경망)
합성곱 신경망(Convolutional Neural Network, CNN)은 최소한의 전처리(preprocess)를 사용하도록 설계된 다계층 [퍼셉트론](https://ko.wikipedia.org/wiki/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0)(multilayer perceptrons)의 한 종류이다. CNN은 하나 또는 여러개의 [합성곱](https://ko.wikipedia.org/wiki/%ED%95%A9%EC%84%B1%EA%B3%B1) 계층과 그 위에 올려진 일반적인 인공 신경망 계층들로 이루어져 있으며, 가중치와 통합 계층(pooling layer)들을 추가로 활용한다. 이러한 구조 덕분에 CNN은 2차원 구조의 입력 데이터를 충분히 활용할 수 있다. 다른 딥 러닝 구조들과 비교해서, CNN은 영상, 음성 분야 모두에서 좋은 성능을 보여준다. CNN은 또한 표준 역전달을 통해 훈련될 수 있다. CNN은 다른 피드포워드 인공신경망 기법들보다 쉽게 훈련되는 편이고 적은 수의 매개변수를 사용한다는 이점이 있다. 최근 딥 러닝에서는 합성곱 심층 신뢰 신경망 (Convolutional Deep Belief Network, CDBN) 가 개발되었는데, 기존 CNN과 구조적으로 매우 비슷해서, 그림의 2차원 구조를 잘 이용할 수 있으며 그와 동시에 심층 신뢰 신경망 (Deep Belief Network, DBN)에서의 선훈련에 의한 장점도 취할 수 있다. CDBN은 다양한 영상과 신호 처리 기법에 사용될 수 있는 일반적인 구조를 제공하며 CIFAR 와 같은 표준 이미지 데이터에 대한 여러 벤치마크 결과에 사용되고 있다. (출처:위키피디아)
### cost function (비용함수)
=error function (오차함수)
## D
### DNN (Deep Neural Network, 심층 신경망)
심층 신경망(Deep Neural Network, DNN)은 입력층(input layer)과 출력층(output layer) 사이에 여러 개의 은닉층(hidden layer)들로 이뤄진 [인공신경망](https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D)(Artificial Neural Network, ANN)이다. 심층 신경망은 일반적인 인공신경망과 마찬가지로 복잡한 [비선형](https://ko.wikipedia.org/wiki/%EB%B9%84%EC%84%A0%ED%98%95) 관계(non-linear relationship)들을 [모델링](https://ko.wikipedia.org/wiki/%EB%AA%A8%EB%8D%B8)할 수 있다. 예를 들어, 사물 식별 모델을 위한 심층 신경망 구조에서는 각 객체가 이미지 기본 요소들의 계층적 구성으로 표현될 수 있다.[18] 이때, 추가 계층들은 점진적으로 모여진 하위 계층들의 특징들을 규합시킬 수 있다. 심층 신경망의 이러한 특징은, 비슷하게 수행된 인공신경망에 비해 더 적은 수의 유닛(unit, node)들 만으로도 복잡한 데이터를 모델링할 수 있게 해준다. (출처:위키피디아)
## E
### error function (오차함수)
선형회귀 방정식에서 weight과 bias를 수정하면서 결과를 얻는데, 반복될 때 얼마나 개선되고 있는지에 대한 측정하는 것 (출처:[텐서플로 첫걸음](https://tensorflowkorea.wordpress.com/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EC%B2%AB%EA%B1%B8%EC%9D%8C/))
## F
## G
### gradient descent (경하 하강법)
함수 값을 최소화하는 최적화 알고리즘
## H
## I
## J
## K
## L
### learning rate (학습속도)
### linear regression (선형회귀분석)
통계학에서, 선형 회귀(線型回歸, 영어: linear regression)는 종속 변수 y와 한 개 이상의 독립 변수 (또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법이다. 한 개의 설명 변수에 기반한 경우에는 단순 선형 회귀, 둘 이상의 설명 변수에 기반한 경우에는 다중 선형 회귀라고 한다.
선형 회귀는 선형 예측 함수를 사용해 회귀식을 모델링하며, 알려지지 않은 파라미터는 데이터로부터 추정한다. 이렇게 만들어진 회귀식을 선형 모델이라고 한다. (출처:위키피디아)

### logistic regression (로지스틱 회귀)
로지스틱 회귀(영어: logistic regression)는 D.R.Cox가 1958년 에 제안한 확률 모델로서 독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법이다.
로지스틱 회귀의 목적은 일반적인 회귀 분석의 목표와 동일하게 종속 변수와 독립 변수간의 관계를 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것이다. 이는 독립 변수의 선형 결합으로 종속 변수를 설명한다는 관점에서는 선형 회귀 분석과 유사하다. 하지만 로지스틱 회귀는 선형 회귀 분석과는 다르게 종속 변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류 (classification) 기법으로도 볼 수 있다. (출처:위키피디아)
## M
## N
## O
## P
### placeholder (플레이스홀더)
프로그램 실행 중에 데이터를 변경하기 위해서 사용하는 '심벌림' 변수.
## Q
## R
### RNN (Recurrent Neural Network, 순환 신경망)
순환 신경망은 [인공신경망](https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D)을 구성하는 유닛 사이의 연결이 Directed cycle을 구성하는 신경망을 말한다. 순환 신경망은 앞먹임 신경망과 달리, 임의의 입력을 처리하기 위해 신경망 내부의 메모리를 활용할 수 있다. 이러한 특성에 의해 순환 신경망은 필기체 인식(Handwriting recognition)과 같은 분야에 활용되고 있고, 높은 인식률을 나타낸다. 순환 신경망을 구성할 수 있는 구조에는 여러가지 방식이 사용되고 있다. 완전 순환망(Fully Recurrent Network), Hopfield Network, Elman Network, Echo state network(ESN), Long short term memory network(LSTM), Bi-directional RNN, Continuous-time RNN(CTRNN), Hierarchical RNN, Second Order RNN 등이 대표적인 예이다. 순환 신경망을 훈련(Training)시키기 위해 대표적으로 [경사 하강법](https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95), Hessian Free Optimization, Global Optimization Methods 방식이 쓰이고 있다. 하지만 순환 신경망은 많은 수의 뉴런 유닛이나 많은 수의 입력 유닛이 있는 경우에 훈련이 쉽지 않은 스케일링 이슈를 가지고있다. (출처:위키피디아)
## S
### Sigmoid (시그모이드)
완만한 S자 형태를 보이는 그래프. sigmoid 함수의 리턴은 0 ~ 1 사이의 값이다.
### Softmax (소프트맥스)
모든 확률을 더했을때 1을 만드는 함수. 예를 들자면 여러 경우의 수를 0.7, 0.2, 0.1 이런식으로 보여준다.
## T
### Tensor (텐서)
동적 크기를 갖는 다차원 데이터 배열로 볼 수 있으며 불리언이나 문자열, 여러 종류의 숫자 같은 정적 자료형을 가지는 텐서플로우의 기본 자료구조 (출처:[텐서플로 첫걸음](https://tensorflowkorea.wordpress.com/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EC%B2%AB%EA%B1%B8%EC%9D%8C/))
### TensorFlow (텐서플로우)
텐서플로우(TensorFlow) 는 구글 제품에 사용되는 머신러닝(기계학습)을 위한 오픈소스 소프트웨어 라이브러리입니다.
## U
## V
## W
## X
## Y
## Z
