# neural_network/neural_network.py
import joblib
import numpy as np

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, learning_rate=0.01, momentum=0.9):
        """
        初始化神经网络参数
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
        self.output_layer_output = self.output_layer_input  # 线性激活

        return self.output_layer_output

    def backward(self, inputs, expected_output):
        """
        反向传播
        :param inputs: 输入数据
        :param expected_output: 期望输出（目标）
        """
        # 计算输出层误差
        output_error = expected_output - self.output_layer_output
        output_delta = output_error  # 线性激活的导数为1

        # 计算隐藏层误差
        hidden_error = output_delta.dot(self.weights_hidden_output.T)  # 错误反向传播到隐藏层
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # 更新权重和偏置（使用动量）
        delta_w_hidden_output_new = np.dot(self.hidden_layer_output.reshape(-1, 1), output_delta.reshape(1, -1)) * self.learning_rate
        self.weights_hidden_output += delta_w_hidden_output_new + self.delta_w_hidden_output * self.momentum
        self.delta_w_hidden_output = delta_w_hidden_output_new

        delta_w_input_hidden_new = np.dot(self.input_layer.reshape(-1, 1), hidden_delta.reshape(1, -1)) * self.learning_rate
        self.weights_input_hidden += delta_w_input_hidden_new + self.delta_w_input_hidden * self.momentum
        self.delta_w_input_hidden = delta_w_input_hidden_new

        # 更新偏置（偏置是一个一维数组，直接求和）
        delta_bias_output_new = np.sum(output_delta, axis=0) * self.learning_rate
        self.bias_output += delta_bias_output_new + self.delta_bias_output * self.momentum
        self.delta_bias_output = delta_bias_output_new

        delta_bias_hidden_new = np.sum(hidden_delta, axis=0) * self.learning_rate
        self.bias_hidden += delta_bias_hidden_new + self.delta_bias_hidden * self.momentum
        self.delta_bias_hidden = delta_bias_hidden_new

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
                # 计算整个训练集的输出
                outputs = np.array([self.forward(x) for x in X_train])
                loss = np.mean(np.square(y_train - outputs))  # 计算当前损失
                print(f"Epoch {epoch}, Loss: {loss}")

    def save_weights(self, filename):
        """
        保存模型权重和偏置到文件
        """
        weights = {
            'weights_input_hidden': self.weights_input_hidden,
            'weights_hidden_output': self.weights_hidden_output,
            'bias_hidden': self.bias_hidden,
            'bias_output': self.bias_output
        }
        joblib.dump(weights, filename)
        print(f"模型权重已保存到 {filename}")

    def load_weights(self, filename):
        """
        从文件加载模型权重和偏置
        """
        weights = joblib.load(filename)
        self.weights_input_hidden = weights['weights_input_hidden']
        self.weights_hidden_output = weights['weights_hidden_output']
        self.bias_hidden = weights['bias_hidden']
        self.bias_output = weights['bias_output']
        print(f"模型权重已从 {filename} 加载")

    def get_last_delta_weights(self):
        """
        获取最后一次权重变化量（delta weights）
        :return: 字典包含所有delta weights
        """
        return {
            'delta_w_input_hidden': self.delta_w_input_hidden,
            'delta_w_hidden_output': self.delta_w_hidden_output,
            'delta_bias_hidden': self.delta_bias_hidden,
            'delta_bias_output': self.delta_bias_output
        }

    def set_delta_weights(self, delta_weights):
        """
        设置权重变化量（delta weights）
        :param delta_weights: 包含所有delta weights的字典
        """
        self.delta_w_input_hidden = delta_weights['delta_w_input_hidden']
        self.delta_w_hidden_output = delta_weights['delta_w_hidden_output']
        self.delta_bias_hidden = delta_weights['delta_bias_hidden']
        self.delta_bias_output = delta_weights['delta_bias_output']

    def apply_delta_weights(self, delta_weights):
        """
        将 delta weights 应用到当前权重和偏置上
        :param delta_weights: 包含所有delta weights的字典
        """
        self.weights_input_hidden += delta_weights['delta_w_input_hidden']
        self.weights_hidden_output += delta_weights['delta_w_hidden_output']
        self.bias_hidden += delta_weights['delta_bias_hidden']
        self.bias_output += delta_weights['delta_bias_output']
        print("Delta weights 已应用到模型中。")
