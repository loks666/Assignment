# NeuralNetHolder.py
import numpy as np
import joblib
import pandas as pd

from neural_network.neural_network import NeuralNetwork  # 确保路径正确


class NeuralNetHolder:
    def __init__(self):
        # 加载缩放器
        self.scaler = joblib.load('neural_network/scaler.pkl')
        print("缩放器已加载")

        # 创建神经网络实例
        self.nn = NeuralNetwork(num_inputs=4, num_hidden_neurons=3, num_outputs=2)

        # 加载训练好的权重
        self.nn.load_weights('neural_network/trained_model.pkl')
        print("训练好的模型权重已加载")

    def predict(self, input_row):
        """
        接收输入行，返回预测的 Vx 和 Vy
        :param input_row: 包含 ['X_dist', 'Y_dist', 'Vx', 'Vy'] 的列表或数组
        :return: [Vx_pred, Vy_pred]
        """
        print(f"原始输入: {input_row}")

        # 确保 input_row 是一个包含四个数值的列表或数组
        if isinstance(input_row, str):
            try:
                # 尝试将字符串拆分为数值
                input_row = [float(x) for x in input_row.split(',')]
                print(f"拆分后的输入: {input_row}")
            except ValueError:
                raise ValueError("输入数据包含非数值字符，无法转换为浮点数")
        elif isinstance(input_row, (list, np.ndarray)):
            input_row = list(input_row)
            print(f"列表/数组输入: {input_row}")
        else:
            raise ValueError("输入数据格式不正确")

        # 确保有四个特征
        if len(input_row) != 4:
            raise ValueError(f"输入特征数量不正确，期望4个，收到{len(input_row)}个")

        # 转换为 DataFrame，保留列名
        input_df = pd.DataFrame([input_row], columns=['X_dist', 'Y_dist', 'Vx', 'Vy'])
        print(f"转换后的 DataFrame: {input_df}")

        # 数据归一化
        input_scaled = self.scaler.transform(input_df)
        print(f"归一化后的输入: {input_scaled}")

        # 前向传播
        output = self.nn.forward(input_scaled[0])
        print(f"神经网络预测输出: {output}")

        # 返回预测结果
        return output.tolist()
