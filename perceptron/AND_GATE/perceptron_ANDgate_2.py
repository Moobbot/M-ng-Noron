# importing Python library
import numpy as np

# define Unit Step Function - Xác định chức năng Bước đơn vị


def unitStep(v):
    return np.where(v >= 0, 1, 0)

# design Perceptron Model


def perceptronModel(x, w, b):
    v = np.dot(w, x) + b  # np.dot() hàm nhân
    y = unitStep(v)
    return y

# AND Logic Function
# w1 = 1, w2 = 1, b = -1.5


def AND_logicFunction(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptronModel(x, w, b)


# testing the Perceptron Model
test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])

print("AND({}, {}) = {}".format(0, 0, AND_logicFunction(test1)))
print("AND({}, {}) = {}".format(0, 1, AND_logicFunction(test2)))
print("AND({}, {}) = {}".format(1, 0, AND_logicFunction(test3)))
print("AND({}, {}) = {}".format(1, 1, AND_logicFunction(test4)))
