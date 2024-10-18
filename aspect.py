import pandas as pd
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load and clean data
site_data = pd.read_excel("data/1_Alldata.xlsx", sheet_name="Plotdata")[['Site', 'PL']]
gee_data = pd.read_csv("data/results_Export.csv")

merged_data = pd.merge(site_data, gee_data, left_on='Site', right_on='site', how='left')
merged_data_cleaned = merged_data.drop(columns=['Site', 'site', 'system:index', '.geo'])

# Convert column names to lowercase
merged_data_cleaned.columns = merged_data_cleaned.columns.str.lower()
# 假设 merged_data_cleaned 已经定义，并且 'pl' 是目标变量
results = {}

# 遍历所有列
for column in merged_data_cleaned.columns:
    if column != 'pl':
        X = merged_data_cleaned[[column]]  # 选择当前列作为自变量
        y = merged_data_cleaned['pl']  # 目标变量

        # 添加常数项
        X_const = sm.add_constant(X)

        # 拟合线性回归模型
        model = sm.OLS(y, X_const).fit()
        r_squared = model.rsquared
        
        # 记录结果
        results[column] = r_squared

# 将结果转为 DataFrame 并输出
results_df = pd.DataFrame(list(results.items()), columns=['Variable', 'R_squared'])
print(results_df)

# 选择 R² 大于 0.3 的自变量
good_fit = results_df[results_df['R_squared'] > 0.3]
print("\n适合线性回归的自变量（R² > 0.3）:\n", good_fit)
