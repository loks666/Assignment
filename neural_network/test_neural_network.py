# neural_network/test_neural_network.py
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

# 确保正确导入 NeuralNetwork 类
from neural_network import NeuralNetwork

def main():
    # 定义文件路径
    scaler_path = 'scaler.pkl'  # scaler.pkl 位于 neural_network 文件夹内
    trained_model_path = 'trained_model.pkl'  # trained_model.pkl 位于 neural_network 文件夹内
    delta_weights_path = 'last_delta_weights.pkl'  # last_delta_weights.pkl 位于 neural_network 文件夹内
    test_features_path = 'test_features.csv'  # 假设 test_features.csv 位于项目根目录
    test_target_path = 'test_target.csv'  # 假设 test_target.csv 位于项目根目录

    # 1. 加载测试集数据
    try:
        X_test_raw = pd.read_csv(test_features_path)
        y_test = pd.read_csv(test_target_path).values
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    # 2. 加载缩放器
    if not os.path.exists(scaler_path):
        print(f"错误: 缩放器文件 '{scaler_path}' 未找到。请确保已运行数据预处理脚本。")
        return

    scaler = joblib.load(scaler_path)
    print("缩放器已加载。")

    # 3. 创建神经网络实例并加载训练好的权重
    nn = NeuralNetwork(num_inputs=4, num_hidden_neurons=5, num_outputs=2, learning_rate=0.01, momentum=0.9)

    if not os.path.exists(trained_model_path):
        print(f"错误: 训练好的模型文件 '{trained_model_path}' 未找到。请确保已运行训练脚本。")
        return

    nn.load_weights(trained_model_path)
    print("训练好的模型权重已加载。")

    # 4. 加载并应用最后一次 Delta Weights
    if not os.path.exists(delta_weights_path):
        print(f"错误: Delta Weights 文件 '{delta_weights_path}' 未找到。请确保已运行训练脚本。")
        return

    last_delta_weights = joblib.load(delta_weights_path)
    nn.apply_delta_weights(last_delta_weights)  # 将 delta weights 应用到模型中
    print("最后一次 Delta Weights 已加载并应用。")

    # 5. 打印和保存权重信息（可选）
    print("\n=== 权重信息 ===")

    print("\nWeights Input to Hidden:")
    print(nn.weights_input_hidden)

    print("\nWeights Hidden to Output:")
    print(nn.weights_hidden_output)

    print("\nBias Hidden:")
    print(nn.bias_hidden)

    print("\nBias Output:")
    print(nn.bias_output)

    # 可选：将权重保存到 CSV 文件
    weights_dir = 'weights_csv_test'
    os.makedirs(weights_dir, exist_ok=True)

    pd.DataFrame(nn.weights_input_hidden, columns=[f'Hidden_{i+1}' for i in range(nn.num_hidden_neurons)]).to_csv(os.path.join(weights_dir, 'weights_input_hidden.csv'), index=False)
    pd.DataFrame(nn.weights_hidden_output, columns=[f'Output_{i+1}' for i in range(nn.num_outputs)]).to_csv(os.path.join(weights_dir, 'weights_hidden_output.csv'), index=False)
    pd.DataFrame(nn.bias_hidden, columns=['Bias_Hidden']).to_csv(os.path.join(weights_dir, 'bias_hidden.csv'), index=False)
    pd.DataFrame(nn.bias_output, columns=['Bias_Output']).to_csv(os.path.join(weights_dir, 'bias_output.csv'), index=False)

    print(f"\n权重已保存到 '{weights_dir}' 文件夹。")

    # 6. 数据归一化
    X_test_scaled = scaler.transform(X_test_raw)
    print("测试集数据已归一化。")

    # 7. 执行前向传播预测
    y_pred = []
    for i in range(len(X_test_scaled)):
        input_data = X_test_scaled[i]
        output = nn.forward(input_data)
        y_pred.append(output)

    y_pred = np.array(y_pred)

    # 8. 计算 MSE、RMSE 和 MAE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== 测试集评估 ===")
    print(f"Mean Squared Error (MSE) on Test Set: {mse}")
    print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse}")
    print(f"Mean Absolute Error (MAE) on Test Set: {mae}")

    # 9. 打印最后一个测试样本的预测结果
    if len(X_test_scaled) > 0:
        last_input_scaled = X_test_scaled[-1]
        y_pred_last = nn.forward(last_input_scaled)
        print(f"\n=== 最后一个测试输入的预测输出 ===\n预测输出: {y_pred_last}")

if __name__ == "__main__":
    main()
