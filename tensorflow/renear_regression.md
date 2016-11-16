# 선형회귀분석(renear regression)

### [텐서플로 첫걸음] 2.1 변수 간의 관계에 대한 모델

```python
import numpy as np

num_points = 1000
vectors_set = []

for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# 데이터를 그림으로 표현
import matplotlib.pyplot as plt # needs "pip install matplotlib"

plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()
```
#### 데이터를 그림으로 표현
<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/renear_regression.png">
</div>
