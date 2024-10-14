import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 读取Excel文件
site_data = pd.read_excel("data/1_Alldata.xlsx", sheet_name="Plotdata")[['Site', 'PL']]
gee_data = pd.read_csv("data/results_Export.csv")

# 2. 合并数据并清洗
merged_data = pd.merge(site_data, gee_data, left_on='Site', right_on='site', how='left')
merged_data_cleaned = merged_data.drop(columns=['Site', 'site', 'system:index', '.geo'])

# 3. 定义特征变量和目标变量
X = merged_data_cleaned.drop(columns=['PL'])  # 特征变量
y = merged_data_cleaned['PL']  # 目标变量

# 计算VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# 初步计算所有特征的VIF值
vif_data = calculate_vif(X)
print("初始VIF值：")
print(vif_data)

# 设定VIF阈值，通常选择10
vif_threshold = 10

# 循环去除VIF值大于阈值的特征
while vif_data['VIF'].max() > vif_threshold:
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
    X = X.drop(columns=[max_vif_feature])
    vif_data = calculate_vif(X)
    print(f"去除特征: {max_vif_feature}, 更新后的VIF值：")
    print(vif_data)

# 保存最终特征及其VIF值
final_vif_data = calculate_vif(X)
final_vif_data.to_csv('data/final_vif_values.csv', index=False)

# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 初始化并训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. 预测并评估模型
y_pred = rf.predict(X_test)

# 7. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# # 输出结果
# print(f"均方误差 (MSE): {mse:.4f}")
# print(f"R² 得分: {r2:.4f}")

# # 保存结果为CSV文件
# results = pd.DataFrame({'Metric': ['MSE', 'R2'], 'Value': [mse, r2]})
# results.to_csv('model_performance_results.csv', index=False)
