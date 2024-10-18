import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 设置字体
site_data = pd.read_excel("data/1_Alldata.xlsx", sheet_name="Plotdata")[['Site', 'PL']]
gee_data = pd.read_csv("data/results_Export.csv")

merged_data = pd.merge(site_data, gee_data, left_on='Site', right_on='site', how='left')
merged_data_cleaned = merged_data.drop(columns=['Site', 'site', 'system:index', '.geo'])

# 将列名转为小写
merged_data_cleaned.columns = merged_data_cleaned.columns.str.lower()

# 假设 merged_data_cleaned 已经定义，并且 'pl' 是目标变量
results = {}

# 遍历所有列
for column in merged_data_cleaned.columns:
    if column != 'pl':
        X = merged_data_cleaned[[column]]  # 选择当前列作为自变量
        y = merged_data_cleaned['pl']  # 目标变量

        # 创建多项式项（这里以二次项为例）
        X_poly = np.column_stack((X, X**2))
        X_poly_const = sm.add_constant(X_poly)  # 添加常数项

        # 拟合多项式回归模型
        model_poly = sm.OLS(y, X_poly_const).fit()
        r_squared_poly = model_poly.rsquared

        # 记录结果
        results[column] = r_squared_poly

# 将结果转为 DataFrame
results_df = pd.DataFrame(list(results.items()), columns=['Variable', 'R_squared'])

# 选择 R² 大于 0.2 的自变量并排序
good_fit_poly = results_df[results_df['R_squared'] > 0.2].sort_values(by='R_squared', ascending=False).reset_index(drop=True)

# 生成 LaTeX 文档
latex_output = r"""\documentclass{article}
\usepackage{booktabs}
\begin{document}
\title{自变量与 R² 值}
\author{}
\date{}
\maketitle

\section*{自变量与 R² 值大于 0.2}

\begin{table}[h]
    \centering
    \begin{tabular}{@{}ll@{}}
        \toprule
        \textbf{自变量} & \textbf{R² 值} \\ \midrule
"""

# 将数据添加到 LaTeX 表格
for index, row in good_fit_poly.iterrows():
    latex_output += f"        {row['Variable']} & {row['R_squared']:.4f} \\\\\n"

latex_output += r"""        \bottomrule
    \end{tabular}
    \caption{R² 值大于 0.2 的自变量}
\end{table}

\end{document}
"""

# 保存为 .tex 文件
with open("data/results.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

print("LaTeX 文档已生成：results.tex")
