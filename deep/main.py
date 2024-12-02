import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# 1. 数据加载
train = pd.read_csv('rossmann-store-sales/train.csv')
test = pd.read_csv('rossmann-store-sales/test.csv')
store = pd.read_csv('rossmann-store-sales/store.csv')

# 合并 Store 数据
train = train.merge(store, on='Store', how='left')
test = test.merge(store, on='Store', how='left')

# 填充缺失值
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# 删除关闭的商店记录
train = train[train['Open'] == 1]

# 转换日期类型并提取时间特征
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

for df in [train, test]:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)

# 对分类变量进行 Label Encoding
categorical_cols = ['StateHoliday', 'StoreType', 'Assortment']
le = LabelEncoder()

for col in categorical_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 定义特征和目标
target = 'Sales'
features = [
    'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'DayOfYear',
    'WeekOfYear', 'Quarter', 'IsWeekend', 'StateHoliday',
    'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance'
]
X = train[features]
y = train[target]
X_test = test[features]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 定义模型
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, activation='relu', random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
# "   MLPRegressor": MLPRegressor(
#         hidden_layer_sizes=(10,),            # 使用一个隐藏层，只有 10 个神经元
#         max_iter=100,                        # 将最大迭代次数减少到 100
#         activation='relu',                   # 使用 ReLU 激活函数
#         random_state=42,
#         early_stopping=True,                 # 启用早停法
#         validation_fraction=0.1,             # 用 10% 的数据进行验证
#         n_iter_no_change=5,                  # 如果 5 轮内验证集上没有变化，则提前停止
#         learning_rate_init=0.01              # 设置初始学习率
#     ),
    "CatBoost": CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=10, verbose=0, random_state=42)
}


# 3. 训练模型并评估
mse_scores = {}
rmse_scores = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mse_scores[name] = mse
    rmse_scores[name] = rmse
    print(f"{name} RMSE: {rmse}")

# 4. 找出最佳模型
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with RMSE: {rmse_scores[best_model_name]}")

# 5. 使用最佳模型生成预测数据
print("Generating predictions with the best model...")
y_test_pred = best_model.predict(X_test)

# 创建提交文件
submission = pd.DataFrame({'Id': test['Id'], 'Sales': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# 6. 输出最佳模型的 RMSE 比例误差图
# 比较每个模型的 RMSE 和相对误差（相对RMSE = RMSE / 平均值）
relative_rmse = {model: rmse / y_train.mean() * 100 for model, rmse in rmse_scores.items()}

# 绘制图形
plt.figure(figsize=(10, 6))
plt.barh(list(relative_rmse.keys()), list(relative_rmse.values()), color='skyblue')
plt.xlabel('Relative RMSE (%)')
plt.title('Model Comparison - Relative RMSE')
plt.tight_layout()
plt.show()
