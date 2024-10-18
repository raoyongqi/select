import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from lime import lime_tabular

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

# 5. 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 创建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # 输出层
])

# 7. 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 8. 训练模型
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# 9. 预测和评估模型
y_pred = model.predict(X_test_scaled)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R²: {r_squared:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# 10. 使用LIME可解释性
explainer = lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=X.columns, mode='regression')

# 11. 选择一条测试数据进行解释
i = 0  # 选择第一条测试数据
exp = explainer.explain_instance(X_test_scaled[i], model.predict, num_features=8)

# 12. 绘制解释结果
exp.show_in_notebook(show_table=True)

# 13. 绘制特征重要性图
plt.figure(figsize=(12, 6))
plt.title('Feature Importance from LIME')
feature_names = [X.columns[j] for j in exp.as_list()[0][0]]
importance_values = [exp.as_list()[j][1] for j in range(len(exp.as_list()))]

plt.barh(feature_names, importance_values)
plt.xlabel('Importance')
plt.show()
