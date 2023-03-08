import numpy as np

# Define the sigmoid function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function


def sigmoid_derivative(x):
    return x * (1 - x)


# Define the inputs and expected outputs for the XOR gate - Bảng giá trị chân lý
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Define the number of nodes in the input, hidden, and output layers - Xác định số lượng nút trong các lớp đầu vào, ẩn và đầu ra
input_nodes = 2
hidden_nodes = 5
output_nodes = 1

# Initialize the weights for the input and hidden layers - Khởi tạo trọng số cho lớp đầu vào và lớp ẩn
hidden_layer_weights = np.random.uniform(size=(input_nodes, hidden_nodes))
output_layer_weights = np.random.uniform(size=(hidden_nodes, output_nodes))

# Set the learning rate
learning_rate = 0.1

# Train the neural network
for epoch in range(10000):
    # Forward propagation
    hidden_layer_activation = sigmoid(np.dot(inputs, hidden_layer_weights))
    output = sigmoid(np.dot(hidden_layer_activation, output_layer_weights))

    # Backpropagation
    output_layer_error = expected_output - output
    output_layer_gradient = output_layer_error * sigmoid_derivative(output)
    hidden_layer_error = output_layer_gradient.dot(
        output_layer_weights.T) * sigmoid_derivative(hidden_layer_activation)
    hidden_layer_gradient = hidden_layer_error * \
        sigmoid_derivative(hidden_layer_activation)

    # Update the weights
    output_layer_weights += hidden_layer_activation.T.dot(
        output_layer_gradient) * learning_rate
    hidden_layer_weights += inputs.T.dot(hidden_layer_gradient) * learning_rate

# Print the output after training
print(output)
