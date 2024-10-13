import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 读取Excel文件
site_data = pd.read_excel("data/1_Alldata.xlsx", sheet_name="Plotdata")[['Site', 'PL']]
gee_data = pd.read_csv("data/results_Export.csv")

# 2. 合并数据并清洗
merged_data = pd.merge(site_data, gee_data, left_on='Site', right_on='site', how='left')
merged_data_cleaned = merged_data.drop(columns=['Site','site', 'system:index', '.geo'])

# 3. 定义特征变量和目标变量
X = merged_data_cleaned.drop(columns=['PL'])  # 特征变量
y = merged_data_cleaned['PL']  # 目标变量

# 初始化数据存储列表，用于记录各轮的VIF值
vif_history = []

# 计算VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# 初步计算所有特征的VIF值
vif_data = calculate_vif(X)
vif_history.append(vif_data.copy())  # 记录第一轮的VIF值

# 设定VIF阈值，通常选择10
vif_threshold = 10

# 循环去除VIF值大于阈值的特征，并保存每轮的VIF结果
while vif_data['VIF'].max() > vif_threshold:
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
    X = X.drop(columns=[max_vif_feature])
    vif_data = calculate_vif(X)
    vif_history.append(vif_data.copy())  # 记录每一轮筛选后的VIF值

# 创建一个DataFrame来保存筛选过程中的VIF变化
rounds = ['第一轮', '第二轮', '第三轮', '第四轮', '第五轮', '第六轮', '第七轮', '第八轮', '第九轮']
max_rounds = min(len(vif_history), len(rounds))  # 确保不会超过可用轮次
output_df = pd.DataFrame()

for i in range(max_rounds):
    current_vif = vif_history[i].set_index('Feature')['VIF']
    output_df[rounds[i]] = current_vif

# 将缺失值填充为 '剔除'
output_df.fillna('剔除', inplace=True)

# 添加变量类型和待选变量列
output_df['变量类型'] = np.where(output_df.index.str.contains('Red|Blue|NIR'), '光谱与指数特征', 
                          np.where(output_df.index.str.contains('Elevation|Slope|Aspect|TPI'), '地形特征', '土壤特征'))
output_df['待选变量'] = output_df.index

# 将列顺序调整
output_df = output_df[['变量类型', '待选变量'] + rounds[:max_rounds]]

# 保存为CSV文件
output_df.to_csv('data/vif_selection_process.csv', index=False)

print("筛选过程已保存到 vif_selection_process.csv 文件中")
