import numpy as np

#  Lớp thuật toán Perceptron

# Perceptron là class định nghĩa một perceptron với các thuộc tính như
# số lượng đầu vào (n_inputs),
# Tốc độ học của mô hình (learning_rate),
# số lần lặp lại (n_iterations),
# trọng số (weights),
# hàm kích hoạt (activation_function),
# hàm đạo hàm của hàm kích hoạt (sigmoid_derivative).


class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
      # Khởi tạo trọng số ngẫu nhiên cho mỗi đầu vào trong khoảng từ -1 đến 1.
      # self.weights = 2 * np.random.random((n_inputs, 1)) - 1
      # Khởi tạo trọng số ngẫu nhiên cho mỗi đầu vào trong khoảng từ 0 đến 1.
        self.weights = np.random.random(n_inputs)
        self.weights_old = self.weights
        # ngưỡng ngẫu nhiên - độ lệch
        self.bias = np.random.rand()
        self.bias_old = self.bias
        # Set the learning rate - tốc độ học - mặc định 0.1
        self.learning_rate = learning_rate

# Hàm sigmoid dùng để tính đầu ra của perceptron dựa trên đầu vào và trọng số hiện tại
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# sigmoid_derivative dùng để tính đạo hàm của hàm sigmoid.
    def sigmoid_derivative(self, x):
        return x * (1 - x)
# hàm kích hoạt cho mô hình perceptron, được sử dụng để tính toán đầu ra từ đầu vào và trọng số.

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

# phương thức predict dùng để dự đoán đầu ra của perceptron dựa trên đầu vào và trọng số hiện tại.
    def predict(self, inputs):
        # U = W*X + w0*x0
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # trả về kết quả đầu ra của perceptron sau khi đi qua hàm kích hoạt.
        return self.activation_function(weighted_sum)


# train là phương thức của perceptron để huấn luyện mô hình.
# cập nhật các trọng số và bias dựa trên
# đầu vào training_inputs, labels và số lần lặp lại (n_iterations).
# Hàm này sử dụng thuật toán Gradient Descent để cập nhật trọng số dựa trên
# sai số giữa đầu ra thực tế và dự đoán, và đạo hàm của hàm kích hoạt.
# Mỗi lần lặp lại sẽ duyệt qua từng đầu vào và cập nhật trọng số cho mô hình.

    def train(self, training_inputs, labels, n_iterations):
        count = 0
        error = 0.1
        while count < n_iterations and error != 0:
            error = 0
            print("Round " + str(count+1) + ":")
            # So sánh giá trị đầu ra tính được với giá trị thực tế
            # Sử dụng hàm zip để lặp qua hai mảng cùng lúc.
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if (prediction != label):
                    # sai số error = lấy hiệu giữa nhãn và giá trị dự đoán.
                    error = label - prediction
                    self.weights += self.learning_rate * error * inputs  # Cập nhật trọng số
                    self.bias += self.learning_rate * error  # Cập nhật bias
                    print("d = " + str(label)+"; y = " +
                          str(prediction) + " => e = "+str(error))
                count += 1
        print('weights = ', self.weights)
        print('bias = ', self.bias)
        print("+------+-------+")


# Hàm kiểm tra đưa ra kết quả của bảng giá trị chân lý


def has_zero(s):

    for c in s:
        if c == 0:
            return 0
    return 1


def has_zero2(s):
    return int(all(s))

# Hàm tạo các chuỗi nhị phân kích thước m


def generate_binary_strings(N):
    binary_strings = []
    for i in range(2**N):
        binary_string = format(i, f"0{N}b")
        binary_strings.append([int(c) for c in binary_string])
    return binary_strings


def generate_binary_strings2(m):
    return [list(map(int, format(i, '0{}b'.format(m)))) for i in range(2**m)]

    # Example usage
m = int(input("Số lượng đầu vào: "))
# m = 3
# Tạo ma trận chứa các trường hợp trong bảng giá trị chân lý
training_inputs = np.array(generate_binary_strings2(m))
labels = []
# kết quả của từng trường hợp 1
for item in training_inputs:
    labels.append(has_zero2(item))

print("+---------------+----------------+")
print(" | AND Truth Table | Result |")
for i in range(0, pow(2, m)):
    print(training_inputs[i], " => ", labels[i])
print("+---------------+----------------+")


perceptron = Perceptron(n_inputs=m)
# print(perceptron.weights)

perceptron.train(training_inputs, labels, n_iterations=1000)

# Test the trained model
test_inputs = training_inputs
for inputs in test_inputs:
    print(inputs, perceptron.predict(inputs))
