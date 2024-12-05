# train_and_evaluate.py
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

# 确保正确导入 NeuralNetwork 类
from neural_network import NeuralNetwork

def main():
    # 1. 加载训练集和测试集
    try:
        X_train = pd.read_csv('train_features.csv').values
        X_test = pd.read_csv('test_features.csv').values
        y_train = pd.read_csv('train_target.csv').values
        y_test = pd.read_csv('test_target.csv').values
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    # 2. 创建神经网络实例，输入为4，隐藏层有5个神经元，输出为2（Vx, Vy）
    nn = NeuralNetwork(num_inputs=4, num_hidden_neurons=5, num_outputs=2, learning_rate=0.01, momentum=0.9)

    # 3. 训练神经网络
    nn.train(X_train, y_train, epochs=2000)  # 增加训练轮数

    # 4. 保存训练好的模型权重
    trained_model_path = 'neural_network/trained_model.pkl'
    os.makedirs('neural_network', exist_ok=True)  # 确保文件夹存在
    nn.save_weights(trained_model_path)

    # 5. 保存最后一次 Delta Weights
    last_delta_weights = nn.get_last_delta_weights()
    delta_weights_path = 'neural_network/last_delta_weights.pkl'
    joblib.dump(last_delta_weights, delta_weights_path)
    print(f"最后一次权重变化量已保存到 '{delta_weights_path}'。")

    # 6. 使用测试集进行评估
    print("\n使用测试集进行评估:")
    y_pred = []  # 用于存储预测结果
    for i in range(len(X_test)):
        inputs = X_test[i]  # 使用 NumPy 数组的索引
        target = y_test[i]  # 使用 NumPy 数组的索引
        output = nn.forward(inputs)  # 前向传播
        y_pred.append(output)  # 将预测结果加入到 y_pred 列表中
        print(f"测试输入: {inputs}, 测试目标: {target}, 预测输出: {output}")

    # 计算均方误差（MSE）和平均绝对误差（MAE）
    y_pred = np.array(y_pred)  # 将预测结果列表转换为 NumPy 数组
    mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
    mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差
    print(f"Mean Squared Error (MSE) on Test Set: {mse}")
    print(f"Mean Absolute Error (MAE) on Test Set: {mae}")

if __name__ == "__main__":
    main()
