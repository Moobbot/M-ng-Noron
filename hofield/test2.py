import numpy as np
import random

# Khai báo số đầu vào và số mẫu (n > m)
n = 4
m = 3
Y = []
for i in range(0, m):
    y = []
    for j in range(0, n):
        y.append(random.choice([1, -1]))
    Y.append(y)
Y = np.array(Y)
print(Y)
