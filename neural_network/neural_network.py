import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)  # 随机初始化权重
        self.bias = np.random.randn()  # 随机初始化偏置
        self.output = 0  # 初始化输出

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid 激活函数

    def feedforward(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias  # 计算加权和
        self.output = self.sigmoid(total_input)  # 激活输出
        return self.output


class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]  # 创建神经元

    def feedforward(self, inputs):
        return [neuron.feedforward(inputs) for neuron in self.neurons]  # 计算每个神经元的输出


class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, learning_rate=0.01, momentum=0.9):
        """
        初始化神经网络参数
        :param num_inputs: 输入层的神经元数量（此处为4：X_dist, Y_dist, Vx, Vy）
        :param num_hidden_neurons: 隐藏层神经元数量
        :param num_outputs: 输出层的神经元数量（此处为2：Vx, Vy）
        :param learning_rate: 学习率
        :param momentum: 动量，用于加速收敛
        """
        self.num_inputs = num_inputs
        self.num_hidden_neurons = num_hidden_neurons
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.momentum = momentum

        # 初始化权重：输入层到隐藏层的权重，隐藏层到输出层的权重
        self.weights_input_hidden = np.random.randn(self.num_inputs, self.num_hidden_neurons)  # 输入到隐藏层
        self.weights_hidden_output = np.random.randn(self.num_hidden_neurons, self.num_outputs)  # 隐藏层到输出层

        # 初始化偏置
        self.bias_hidden = np.random.randn(self.num_hidden_neurons)
        self.bias_output = np.random.randn(self.num_outputs)

        # 初始化权重变化量（用于动量）
        self.delta_w_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.delta_w_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.delta_bias_hidden = np.zeros_like(self.bias_hidden)
        self.delta_bias_output = np.zeros_like(self.bias_output)

    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid 导数，用于反向传播"""
        return x * (1 - x)

    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据
        :return: 输出层的结果
        """
        # 输入到隐藏层
        self.input_layer = inputs
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # 隐藏层到输出层
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backward(self, inputs, expected_output):
        """
        反向传播
        :param inputs: 输入数据
        :param expected_output: 期望输出（目标）
        """
        # 计算输出层误差
        output_error = expected_output - self.output_layer_output
        output_delta = output_error * self.sigmoid_derivative(self.output_layer_output)

        # 计算隐藏层误差
        hidden_error = output_delta.dot(self.weights_hidden_output.T)  # 错误反向传播到隐藏层
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # 更新权重和偏置（使用动量）
        # 修正矩阵形状，使得矩阵乘法能够正常进行
        self.weights_hidden_output += np.dot(self.hidden_layer_output.reshape(-1, 1), output_delta.reshape(1,
                                                                                                           -1)) * self.learning_rate + self.delta_w_hidden_output * self.momentum
        self.weights_input_hidden += np.dot(self.input_layer.reshape(-1, 1), hidden_delta.reshape(1,
                                                                                                  -1)) * self.learning_rate + self.delta_w_input_hidden * self.momentum

        # 更新偏置（偏置是一个一维数组，直接求和）
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate + self.delta_bias_output * self.momentum
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate + self.delta_bias_hidden * self.momentum

        # 保存权重变化量，用于下一轮的动量更新
        self.delta_w_input_hidden = np.dot(self.input_layer.reshape(-1, 1),
                                           hidden_delta.reshape(1, -1)) * self.learning_rate
        self.delta_w_hidden_output = np.dot(self.hidden_layer_output.reshape(-1, 1),
                                            output_delta.reshape(1, -1)) * self.learning_rate
        self.delta_bias_hidden = np.sum(hidden_delta, axis=0) * self.learning_rate
        self.delta_bias_output = np.sum(output_delta, axis=0) * self.learning_rate

    def train(self, X_train, y_train, epochs):
        """
        训练神经网络
        :param X_train: 训练集特征
        :param y_train: 训练集目标
        :param epochs: 训练轮数
        """
        for epoch in range(epochs):
            for i in range(len(X_train)):
                inputs = X_train[i]  # 使用 NumPy 数组的索引
                expected_output = y_train[i]  # 使用 NumPy 数组的索引
                self.forward(inputs)  # 前向传播
                self.backward(inputs, expected_output)  # 反向传播

            # 每100个epoch打印一次损失值
            if epoch % 100 == 0:
                loss = np.mean(np.square(y_train - self.forward(X_train)))  # 计算当前损失
                print(f"Epoch {epoch}, Loss: {loss}")
