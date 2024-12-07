import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
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

# 删除测试集中不存在的列
if 'Customers' in train.columns:
    train.drop(['Customers'], axis=1, inplace=True)

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

# 2. 特征筛选 - 使用 CatBoost 的特征重要性
catboost_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, random_state=42, verbose=0)
catboost_model.fit(X, y)
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': catboost_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# 筛选重要性较高的特征
selected_features = feature_importances[feature_importances['Importance'] > 0.01]['Feature'].tolist()
print("\nSelected Features:", selected_features)

X = X[selected_features]
X_test = X_test[selected_features]

# 3. 平滑处理 - 使用指数加权平滑
alpha = 0.3  # 平滑因子
y_smoothed = y.ewm(alpha=alpha, adjust=False).mean()

# 4. 超参数优化和重新训练模型
optimized_model = CatBoostRegressor(
    iterations=1000,  # 增加迭代次数
    learning_rate=0.03,  # 较小的学习率以提高稳定性
    depth=8,  # 限制树的深度，避免过拟合
    random_state=42,
    verbose=100  # 输出训练进度
)
print("\nRetraining CatBoost with optimized parameters...")
optimized_model.fit(X, y_smoothed)

# 5. 在测试集上生成预测
y_test_pred = optimized_model.predict(X_test)

# 6. 创建提交文件
submission_file = "submission_best_catboost.csv"
submission = pd.DataFrame({'Id': test['Id'], 'Sales': y_test_pred})
submission.to_csv(submission_file, index=False)
print(f"\nRefined submission file created: {submission_file}")
