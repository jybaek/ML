# 함수 
대부분(현재는 100%) 내용이 [텐서플로 첫걸음](https://tensorflowkorea.wordpress.com/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EC%B2%AB%EA%B1%B8%EC%9D%8C/)으로 부터 발췌되었습니다.

## 수학 함수
함수 | 설명
--- | ---
tf.add | 덧셈
tf.sub | 뺄셈
tf.mul | 곱셈
tf.div | 나눗셈의 몫
tf.mod | 나눗셈의 나머지
tf.abs | 절댓값을 리턴합니다.
tf.neg | 음수를 리턴합니다.
tf.sign | 부호를 리턴합니다. (음수는 -1, 양수는 1, 0일 땐 0을 리턴)
tf.inv | 역수를 리턴합니다. (예를 들어 3의 역수는 1/3입니다.)
tf.square | 제곱을 계산합니다.
tf.round | 반올림 값을 리턴합니다.
tf.sqrt | 제곱근을 계산합니다.
tf.pow | 거듭제곱 값을 계산합니다.
tf.exp | 지수 값을 계산합니다.
tf.log | 로그 값을 계산합니다.
tf.maximum | 최댓값을 리턴합니다.
tf.minimum | 최솟값을 리턴합니다.
tf.cos | 코사인 함수 값을 계산합니다.
tf.sin | 사인 함수 값을 계산합니다.

## 행렬 연산 함수
함수 | 설명
--- | ---
tf.diag | 대각행렬을 리턴합니다.
tf.transpose | 전치행렬을 리턴합니다.
tf.matmul | 두 텐서를 행렬곱한 결과 텐서를 리턴합니다.
tf.matrix_determinant | 정방행렬의 행렬식 값을 리턴합니다.
tf.matrix_inverse | 정방행렬의 역행렬을 리턴합니다.

## 주요 변환 함수
함수 | 설명
--- | ---
tf.shape | 텐서의 구조를 알아냅니다.
tf.size | 텐서의 크기를 알아냅니다.
tf.rank | 텐서의 랭크를 알아냅니다.
tf.reshape | 텐서의 원소는 그대로 유지하면서 텐서의 구조를 바꿉니다.
tf.squeeze | 텐서에서 크기가 1인 차원을 삭제합니다.
tf.expand_dims | 텐서에 차원을 추가합니다.
tf.slice | 텐서의 일부분을 삭제합니다.
tf.split | 텐서를 한 차원을 기준으로 여러 개의 텐서로 나눕니다.
tf.tile | 한 텐서를 여러 번 중복해서 늘려 새 텐서를 만듭니다.
tf.concat | 한 차원을 기준으로 텐서를 이어 붙입니다.
tf.reverse | 텐서의 지정된 차원을 역전시킵니다.
tf.transpose | 텐서를 전치합니다.
tf.gather | 주어진 인덱스에 따라 텐서의 원소를 모읍니다.

## 상수를 생성하는 다양한 방법
함수 | 설명
--- | ---
tf.zeros_like | 모든 원소를 0으로 초기화한 텐서를 생성합니다.
tf.ones_like | 모든 원소를 1로 초기화한 텐서를 생성합니다.
tf.fill | 주어진 스칼라 값으로 원소를 초기화한 텐서를 생성합니다.
tf.constant | 함수 인수로 지정된 값을 이용하여 상수 텐서를 생성합니다.

## 텐서 생성 관련 함수
함수 | 설명
--- | ---
tf.random_normal | 정규분포를 따르는 난수로 텐서를 생성합니다.
tf.truncated_normal | 정규분포를 따르는 난수로 텐서를 생성하되, 크기가 표준편차의 2배 수보다 큰 값은 제거합니다.
tf.random_uniform | 균등분포를 따르는 난수로 텐서를 생성합니다.
tf.random_shuffle | 첫 번째 차원을 기준으로 텐서의 원소를 섞습니다.
tf.set_random_seed | 난수 시드(seed)를 설정합니다.

## 차원을 감소시키는 수학 연산
함수 | 설명
--- | ---
tf.reduce_sum | 지정한 차원을 따라 원소들을 더합니다.
tf.reduce_prod | 지정한 차원을 따라 원소들을 곱합니다.
tf.reduce_min | 지정한 차원을 따라 최솟값을 계산합니다.
tf.reduce_max | 지정한 차원을 따라 최댓값을 계산합니다.
tf.reduce_mean  지정한 차원을 따라 평균을 계산합니다.

## argmin과 argmax
함수 | 설명
--- | ---
tf.argmin | 지정한 차원을 따라 가장 작은 값의 원소가 있는 인덱스를 리턴합니다.
tf.argmax | 지정한 차원을 따라 가장 큰 값의 원소가 있는 인덱스를 리턴합니다.
