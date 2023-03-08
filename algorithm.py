
# importing Python library
import numpy as np
n = int(input("Số lượng đầu vào: "))
for x in range(n):
    # Random từ 0-10, 1 hàng n kí tự
    print(np.random.randint(10, size=(1, n)))
