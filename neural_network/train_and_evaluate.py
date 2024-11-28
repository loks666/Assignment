import pandas as pd
from sklearn.metrics import mean_squared_error
from neural_network import NeuralNetwork
import numpy as np  # 导入 NumPy

# 1. 加载训练集和测试集
X_train = pd.read_csv('train_features.csv')
X_test = pd.read_csv('test_features.csv')
y_train = pd.read_csv('train_target.csv')
y_test = pd.read_csv('test_target.csv')

# 2. 创建神经网络实例，输入为4，隐藏层有3个神经元，输出为2（Vx, Vy）
nn = NeuralNetwork(num_inputs=4, num_hidden_neurons=3, num_outputs=2)

# 3. 训练神经网络
nn.train(X_train.values, y_train.values, epochs=1000)  # 使用训练集数据进行训练

# 4. 使用测试集进行评估
print("\n使用测试集进行评估:")
y_pred = []  # 用于存储预测结果
for i in range(len(X_test)):
    inputs = X_test.iloc[i].values  # 获取测试集的输入数据
    target = y_test.iloc[i].values  # 获取测试集的目标数据
    output = nn.forward(inputs)  # 前向传播
    y_pred.append(output)  # 将预测结果加入到 y_pred 列表中
    print(f"测试输入: {inputs}, 测试目标: {target}, 预测输出: {output}")

# 计算均方误差（MSE）
y_pred = np.array(y_pred)  # 将预测结果列表转换为 NumPy 数组
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
print(f"Mean Squared Error (MSE) on Test Set: {mse}")
