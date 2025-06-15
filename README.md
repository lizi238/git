# git
大数据
import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('High-Frequency Stock Index Markets NEW.csv')

# 2. 列名清洗：去除多余空格和换行符，统一格式
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace(' ', '_')

# 3. 数据类型转换：将价格和金额相关列转为float
num_cols = ['Opening_Price', 'Highest_Price', 'Lowest_Price', 'Closing_Price', 'Transaction_amount']
for col in num_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

# 4. 数据概况
shape = df.shape
dtypes = df.dtypes
missing = df.isnull().sum()
missing_ratio = (missing / len(df)).round(4)
desc = df.describe(include='all')

# 5. 数据质量评估
quality_report = pd.DataFrame({
    '数据类型': dtypes,
    '缺失值数量': missing,
    '缺失比例': missing_ratio
})

# 6. 数据清洗
# 删除缺失比例>30%的列
drop_cols = quality_report[quality_report['缺失比例'] > 0.3].index.tolist()
df_clean = df.drop(columns=drop_cols)

# 其余缺失值：数值型用中位数，分类型用众数填充
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype in [np.float64, np.int64]:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

# 异常值处理：3倍IQR法，数值型列
outlier_info = {}
for col in df_clean.select_dtypes(include=[np.number]).columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    outlier_info[col] = int(outliers)
    df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
    df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])

# 7. 初步统计分析
stat_summary = df_clean.describe().T

# 8. 保存清洗后数据
df_clean.to_csv('Drivers/High-Frequency_Stock_Index_Markets_NEW_cleaned.csv', index=False)

# 9. 生成报告
with open('Drivers/data_overview_and_cleaning_report_high_freq_stock.md', 'w', encoding='utf-8') as f:
    f.write('# 数据概况与清洗报告\n\n')
    f.write('## 1. 数据集基本信息\n')
    f.write(f'- 数据维度: {shape}\n')
    f.write(f'- 列名: {list(df.columns)}\n\n')
    f.write('## 2. 数据质量评估\n')
    f.write('```\n')
    f.write(quality_report.to_string())
    f.write('\n```\n')
    f.write(f'- 删除缺失比例超过30%的列: {drop_cols}\n\n')
    f.write('## 3. 数据清洗方法说明\n')
    f.write('- 列名标准化，数值型字段转换\n')
    f.write('- 删除高缺失列，其余缺失值用中位数/众数填充\n')
    f.write('- 数值型字段用3倍IQR法截断异常值\n\n')
    f.write('## 4. 异常值处理统计\n')
    f.write('```\n')
    f.write(str(outlier_info))
    f.write('\n```\n')
    f.write('## 5. 清洗后数据初步统计分析\n')
    f.write('```\n')
    f.write(stat_summary.to_string())
    f.write('\n```\n')
    f.write('- 清洗后数据已保存为 Drivers/High-Frequency_Stock_Index_Markets_NEW_cleaned.csv\n')
    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取清洗后的数据
df = pd.read_csv('Drivers/High-Frequency_Stock_Index_Markets_NEW_cleaned.csv')

# 设置中文字体（如有需要可调整）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 收盘价分布直方图
plt.figure(figsize=(8,5))
sns.histplot(df['Closing_Price'], bins=50, kde=True)
plt.title('收盘价分布直方图')
plt.xlabel('收盘价')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig('Drivers/plot1_closing_price_hist.png')
plt.close()

# 2. 成交量箱线图
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Volume'])
plt.title('成交量箱线图')
plt.xlabel('成交量')
plt.tight_layout()
plt.savefig('Drivers/plot2_volume_box.png')
plt.close()

# 3. 不同股票代码的收盘价均值柱状图（前10代码）
top_codes = df['Code'].value_counts().index[:10]
mean_close = df[df['Code'].isin(top_codes)].groupby('Code')['Closing_Price'].mean().sort_values()
plt.figure(figsize=(10,6))
mean_close.plot(kind='barh', color='skyblue')
plt.title('不同股票代码的收盘价均值（前10）')
plt.xlabel('收盘价均值')
plt.ylabel('股票代码')
plt.tight_layout()
plt.savefig('Drivers/plot3_code_closing_price_bar.png')
plt.close()

# 4. 收盘价与成交量的散点图
plt.figure(figsize=(8,5))
sns.scatterplot(x='Closing_Price', y='Volume', data=df, alpha=0.3)
plt.title('收盘价与成交量的关系')
plt.xlabel('收盘价')
plt.ylabel('成交量')
plt.tight_layout()
plt.savefig('Drivers/plot4_closing_vs_volume_scatter.png')
plt.close()

# 5. 主股票代码收盘价时间序列（取样本最多的代码）
main_code = df['Code'].value_counts().idxmax()
df_main = df[df['Code'] == main_code].copy()
# 确保时间字段为字符串类型且可排序
df_main['time'] = pd.to_datetime(df_main['time'], errors='coerce')
df_main = df_main.dropna(subset=['time'])
df_main = df_main.sort_values('time')
plt.figure(figsize=(12,5))
plt.plot(df_main['time'], df_main['Closing_Price'])
plt.title(f'{main_code} 收盘价时间序列')
plt.xlabel('时间')
plt.ylabel('收盘价')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Drivers/plot5_main_code_time_series.png')
plt.close()
