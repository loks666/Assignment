# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存缩放器

# 读取数据，不指定分隔符，保持原样
data = pd.read_csv('ce889_dataCollection.csv', header=None, names=['X_dist', 'Y_dist', 'Vx', 'Vy'])

# 检查是否有缺失值
missing_values = data.isnull().sum()
print("缺失值情况：")
print(missing_values)

# 如果有缺失值，可以选择删除或填充
data = data.dropna()  # 简单删除包含缺失值的行

# 确保目标点坐标正确（根据您的游戏设置）
# 请根据实际游戏中的着陆点坐标进行调整
X_target = 433.0  # 示例目标点 X 坐标，请根据实际情况修改
Y_target = 430.0  # 示例目标点 Y 坐标，请根据实际情况修改

# 计算距离到目标点
data['X_dist'] = data['X_dist'] - X_target
data['Y_dist'] = data['Y_dist'] - Y_target

# 选择特征和目标
# 特征：X_dist, Y_dist, Vx, Vy
# 目标：Vx, Vy（假设您要预测下一个时刻的速度）
X = data[['X_dist', 'Y_dist', 'Vx', 'Vy']]
y = data[['Vx', 'Vy']]

# 分割数据集，80% 训练，20% 测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据归一化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存缩放器
joblib.dump(scaler, 'scaler.pkl')  # 修正保存路径

# 保存训练集和测试集的特征数据
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('train_features.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('test_features.csv', index=False)

# 保存训练集和测试集的目标数据
y_train.to_csv('train_target.csv', index=False)
y_test.to_csv('test_target.csv', index=False)

print("数据已保存为训练集和测试集 CSV 文件，缩放器已保存为 scaler.pkl")
