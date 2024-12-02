import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import lightgbm as lgb
print(lgb.__version__)

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

# 2. 数据预处理
# 删除关闭的商店记录
train = train[train['Open'] == 1]

# 转换日期类型并提取时间特征
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

for df in [train, test]:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)

# 编码分类变量
categorical_cols = ['StateHoliday', 'StoreType', 'Assortment']
for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 目标值和特征
target = 'Sales'
features = [
    'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'DayOfYear',
    'WeekOfYear', 'Quarter', 'IsWeekend', 'StateHoliday',
    'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance'
]
X_train = train[features]
y_train = train[target]
X_test = test[features]

# 3. 模型训练
# 划分验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# LightGBM 数据集
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val, label=y_val)

# 模型参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
}

# 模型训练
model = lgb.train(params, train_data, valid_sets=[train_data, val_data])

# 验证集预测
val_pred = model.predict(X_val, num_iteration=model.best_iteration)

# RMSPE 计算
def rmspe(y, y_pred):
    return np.sqrt(np.mean(((y - y_pred) / y) ** 2))

print("Validation RMSPE:", rmspe(y_val, val_pred))

# 4. 预测与提交
# 测试集预测
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 创建提交文件
submission = pd.DataFrame({'Id': test['Id'], 'Sales': y_test_pred})
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")