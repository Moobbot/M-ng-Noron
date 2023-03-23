import numpy as np


class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        # ma trận trọng số của mạng Hopfield, được khởi tạo ban đầu bằng ma trận 0 với kích thước là n x n.
        self.weights = np.zeros((n, n))
        self.threshold = 0

    # Huấn luyện mạng Hopfield với các mẫu đầu vào
    def train(self, X):
        # Tính toán ma trận trọng số
        for x in X:
            # Công thức tính ma trận trọng số W.
        # Tính tích vô hướng giữa x với chính nó để tính ma trận đối xứng của nó, sau đó cộng vào ma trận trọng số W.
            self.weights += np.outer(x, x)
        # Xóa tất cả các phần tử trên đường chéo chính của ma trận W để giữ cho chúng bằng 0.
        np.fill_diagonal(self.weights, 0)
        
    # Dự đoán kết quả cho một mẫu đầu vào
    def predict(self, Y_train, Y_test):
        U = np.dot(self.weights, Y_test) - self.threshold
        y_out = np.where(U == 0, Y_train, np.where(U >= 0, 1, -1))
        return y_out
        
# Ví dụ với n=10 và 2 mẫu Y1 và Y2
n = int(input("Số đầu vào: "))
m = n
print("Số mẫu nhỏ hơn số đầu vào")
while(m >= n):
    m = int(input("Số mẫu vào: "))
Y = []
for i in range(0, m):
    y = []
    for j in range(0, n):
        y.append(np.random.choice([1, -1]))
    Y.append(y)
Y = np.array(Y)
print("Ma trận mẫu:")
print(Y)
print("-----------")

# Khởi tạo mạng Hopfield
hopfield_net = HopfieldNetwork(n)
    
# Huấn luyện mạng Hopfield với 2 mẫu
hopfield_net.train(Y)

# Dự ma trận trọng số
print(hopfield_net.weights)

# Tạo mẫu thử ngẫu nhiên
X = []

for i in range(n):
    X.append(np.random.choice([1, -1]))
X = np.array(X)
print("----------")
print("Mẫu thử:", X)
print("----------")
    
# Cập nhật đầu vào
for i in range(m):
    print("Ma trận so sánh:", Y[i])
    print("----------")
    print("Ma trận dự đoán "+str(i+1)+": ", hopfield_net.predict(Y[i], X))
    print("----------")
