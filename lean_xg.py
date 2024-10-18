import pandas as pd
import numpy as np
import xgboost as xgb
import shap
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

# 5. 创建 XGBoost 回归模型并拟合
model = xgb.XGBRegressor(random_state=42, missing=np.nan)
model.fit(X_train, y_train)

# 6. 预测和评估模型
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R²: {r_squared:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# 7. 计算 SHAP 值
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# 8. 绘制 SHAP 值
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test)

# 1. 找到最重要的两个特征
importance = np.abs(shap_values.values).mean(axis=0)  # 计算每个特征的平均绝对 SHAP 值
indices = np.argsort(importance)[::-1]  # 从高到低排序特征

# 2. 绘制前两个特征的依赖图
plt.figure(figsize=(12, 6))
shap.dependence_plot(ind=indices[0], shap_values=shap_values.values, features=X_test, interaction_index=indices[1])
plt.title(f'Dependence Plot of {X.columns[indices[0]]} with Interaction of {X.columns[indices[1]]}')
plt.show()

plt.figure(figsize=(12, 6))
shap.dependence_plot(ind=indices[1], shap_values=shap_values.values, features=X_test, interaction_index=indices[0])
plt.title(f'Dependence Plot of {X.columns[indices[1]]} with Interaction of {X.columns[indices[0]]}')
plt.show()