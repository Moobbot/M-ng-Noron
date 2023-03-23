import numpy as np

# Tính ma trận trọng số
def hopfield_weights(X):
    n = X.shape[1]  # số chiều của mẫu huấn luyện
    # ma trận trọng số của mạng Hopfield, được khởi tạo ban đầu bằng ma trận 0 với kích thước là n x n.
    W = np.zeros((n, n))
    for x in X:
        # Công thức tính ma trận trọng số W.
        # Tính tích vô hướng giữa x với chính nó để tính ma trận đối xứng của nó, sau đó cộng vào ma trận trọng số W.
        W += np.outer(x, x)
    # Điều này sẽ xóa tất cả các phần tử trên đường chéo chính của ma trận W để giữ cho chúng bằng 0.
    # để đảm bảo rằng mỗi đầu vào được ánh xạ thành chính nó (trong trường hợp này là 0) và
    # không được gán trọng số khác với bất kỳ đầu vào khác.
    np.fill_diagonal(W, 0)
    return W

# Tính toán kết quả dự đoán
def hopfield_Y(W, Y_train, Y_test, n):
    U = np.dot(W, Y_test) # U = W*Y_test
    for i in range(n):
        if (U[i] == 0):
            U[i] = Y_train[i]
    y_out = np.where(U > 0, 1, -1)
    return y_out

# Khai báo số đầu vào và số mẫu
n = 3
m = 2

# Khởi tạo mẫu
Y = np.array([[1] * n, [-1] * n])
# Tính ma trận trọng số
W = hopfield_weights(Y)
# print(W)
# Thử X = Y => CM khả năng nhớ và nhận dạng
# X = Y.transpose()

# Khởi tạo đầu vào và thêm nhiễu
X = np.array([-1, -1, 1])
# So sánh với đầu vào gốc
print("Input patterns:")
print(X.transpose())
# Cập nhật đầu vào
for i in range(m):
    y_out = hopfield_Y(W, Y[i], X, n)
    print("Output " + str(i) + " patterns:")
    print(y_out)
    

