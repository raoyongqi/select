import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import joblib  # 用于保存模型
import os  # 用于处理文件和目录

# 1. 读取Excel文件
# 读取数据
site_data = pd.read_excel("data/1_Alldata.xlsx", sheet_name="Plotdata")[['Site', 'PL']]
gee_data = pd.read_csv("data/results_Export.csv")

# 使用左连接合并数据
merged_data = pd.merge(site_data, gee_data, left_on='Site',right_on='site', how='left')

# 删除 'system:index' 和 '.geo' 列
merged_data_cleaned = merged_data.drop(columns=['Site','system:index', '.geo'])



# 定义特征变量和目标变量
X = merged_data_cleaned.drop(columns=['PL'])  # 特征变量
y = merged_data_cleaned['PL']  # 目标变量
# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # 5. 初始化并训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. 特征选择
selector = SelectFromModel(rf, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 7. 训练基于选择特征的模型
rf_selected = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# 8. 预测并评估模型
y_pred = rf_selected.predict(X_test_selected)

# 9. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 得分: {r2:.4f}")
