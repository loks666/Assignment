# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存缩放器

# 读取数据
data = pd.read_csv('ce889_dataCollection.csv', header=None, names=['X_dist', 'Y_dist', 'Vx', 'Vy'])

# 数据分割：分为特征和目标
X = data[['X_dist', 'Y_dist', 'Vx', 'Vy']]
y = data[['Vx', 'Vy']]

# 80% 的数据用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据归一化（仅在训练数据上拟合缩放器）
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存缩放器
joblib.dump(scaler, 'scaler.pkl')

# 保存训练集和测试集的特征数据
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('train_features.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('test_features.csv', index=False)

# 保存训练集和测试集的目标数据
y_train.to_csv('train_target.csv', index=False)
y_test.to_csv('test_target.csv', index=False)

print("数据已保存为训练集和测试集 CSV 文件，缩放器已保存为 scaler.pkl")
