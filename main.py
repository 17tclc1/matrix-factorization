import numpy as np
from MF import MF
data = np.array([
  [1.0, 4.0, 5.0, 0, 3.0],
  [5.0, 1.0, 0, 5.0, 2.0],
  [4.0, 1.0, 2.0, 5.0, 0],
  [0, 3.0, 4.0, 0, 4.0]
])
# training using mf
mf = MF(data=data, K_feature=2, beta=0.002, lambda_value=0.01, iterations=20000)
mf.train()
print(mf.gradient_descent())