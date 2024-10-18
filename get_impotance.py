import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 读取Excel文件
site_data = pd.read_excel("data/1_Alldata.xlsx", sheet_name="Plotdata")[['Site', 'PL']]
gee_data = pd.read_csv("data/results_Export.csv")

# 2. 合并数据并清洗
merged_data = pd.merge(site_data, gee_data, left_on='Site', right_on='site', how='left')
merged_data_cleaned = merged_data.drop(columns=['Site', 'site', 'system:index', '.geo'])

# 3. 定义特征变量和目标变量
X = merged_data_cleaned.drop(columns=['PL'])  # 特征变量
y = merged_data_cleaned['PL']  # 目标变量

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 创建随机森林回归模型并拟合
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. 预测和评估模型
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R²: {r_squared:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# 7. 特征重要性
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

# 8. 绘制前8个特征的重要性图
plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(8), importance[indices][:8], align='center')
plt.xticks(range(8), [X.columns[i] for i in indices[:8]], rotation=45)
plt.xlim([-1, 8])
plt.ylabel('Importance')
plt.show()
