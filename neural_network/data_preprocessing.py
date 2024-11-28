import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv('ce889_dataCollection.csv', header=None, names=['X_dist', 'Y_dist', 'Vx', 'Vy'])

# 数据归一化
scaler = MinMaxScaler()
data[['X_dist', 'Y_dist', 'Vx', 'Vy']] = scaler.fit_transform(data[['X_dist', 'Y_dist', 'Vx', 'Vy']])

# 数据分割：分为训练集和测试集
X = data[['X_dist', 'Y_dist', 'Vx', 'Vy']]
y = data[['X_dist', 'Y_dist']]

# 80% 的数据用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 保存训练集和测试集的特征数据
X_train.to_csv('train_features.csv', index=False)
X_test.to_csv('test_features.csv', index=False)

# 保存训练集和测试集的目标数据
y_train.to_csv('train_target.csv', index=False)
y_test.to_csv('test_target.csv', index=False)

print("数据已保存为训练集和测试集 CSV 文件")