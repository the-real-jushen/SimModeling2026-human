# %% [markdown]
"""
# 数据清洗与预处理

数据清洗是数据科学工作流程中最耗时但也最重要的一步。俗话说 "Garbage in, garbage out"（垃圾进，垃圾出），如果输入模型的数据质量很差，那么模型的输出结果也就毫无意义。

**本节课你将学到：**

1.  如何识别和处理缺失值（NaN）。
2.  如何识别和删除重复的行。
3.  如何对 DataFrame 进行排序。

"""

# %% [markdown]
"""
## 1. 处理缺失值 (Missing Values)

我们继续使用上一节的 `health_data.csv`。如果你还记得，我们在最后一行故意留下了一个缺失的 `Calories` 值。

"""

# %%
import pandas as pd
import numpy as np

# 重新读取数据
data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/health_data.csv'
df = pd.read_csv(data_path)

print("--- 原始数据 (注意最后一行的 Calories 是 NaN) ---")
print(df.tail(3))

# 策略 A：删除包含任何缺失值的行
# dropna() 默认会删除任何包含 NaN 的行。
# inplace=True 表示直接在原 DataFrame 上进行修改，而不是返回一个新的 DataFrame。
df_dropped = df.copy() # 复制一份以免影响后续演示
df_dropped.dropna(inplace=True)
print("\n--- 删除缺失值后的数据 (最后一行被删除了) ---")
print(df_dropped.tail(3))

# 策略 B：填充缺失值
# fillna() 可以用指定的值替换 NaN。
df_filled = df.copy()
# 我们用 'Calories' 这一列的平均值来填充缺失的部分
mean_calories = df_filled['Calories'].mean()
print(f"\nCalories 的平均值为: {mean_calories:.2f}")

df_filled['Calories'].fillna(mean_calories, inplace=True)
print("--- 填充平均值后的数据 (最后一行被填上了平均值) ---")
print(df_filled.tail(3))




# %% [markdown]
"""
## 2. 处理重复值 (Duplicates)

为了演示，我们先人为地向 DataFrame 中添加几行重复的数据。

"""

# %%
# 人为添加重复行
# 我们把第 0 行和第 1 行复制并追加到末尾
df_with_dups = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

print("--- 包含重复行的数据 (注意最后两行与前两行相同) ---")
print(df_with_dups.head(2))
print("...")
print(df_with_dups.tail(2))

# 1. 检查重复行
# duplicated() 返回一个布尔型 Series，True 表示该行是前面出现过的重复行
duplicates_mask = df_with_dups.duplicated()
print(f"\n发现 {duplicates_mask.sum()} 行重复数据。")

# 2. 删除重复行
# drop_duplicates() 会保留第一次出现的记录，删除后续的重复记录
df_with_dups.drop_duplicates(inplace=True)

print("\n--- 删除重复行后的数据信息 ---")
df_with_dups.info() # 行数应该恢复到原来的 10 行




# %% [markdown]
"""
## 3. 数据排序 (Sorting)

使用 `sort_values()` 方法可以轻松地对 DataFrame 进行排序。

"""

# %%
# 按照 'Pulse' (心率) 列进行升序排序
df_sorted_asc = df.sort_values(by='Pulse')
print("--- 按心率升序排序 (前 5 行) ---")
print(df_sorted_asc.head())

# 按照 'Calories' (卡路里) 列进行降序排序
# ascending=False 表示降序
df_sorted_desc = df.sort_values(by='Calories', ascending=False)
print("\n--- 按卡路里降序排序 (前 5 行) ---")
print(df_sorted_desc.head())




# %% [markdown]
"""
### 综合练习：探索 311 投诉数据

现在让我们用一个小测验来检验你对 Pandas 索引和过滤的掌握程度。在这个小练习中，你将读取并探索纽约市的 311 投诉数据集 (`311-service-requests.csv`)。

你需要完成三个任务：
1. 找出总体上最多的抱怨类型。
2. 找出哪个区（Borough）发生 "Noise - Street/Sidewalk" 抱怨的次数最多。
3. 找出 "BROOKLYN" 区所有包含 "Noise" 字符串的抱怨记录的数量。

"""

# %%
import pandas as pd
import os

# 获取数据文件路径
data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/311-service-requests.csv'
if not os.path.exists(data_path):
    data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/311-service-requests.csv'
    
# 读取311服务请求数据
complaints = pd.read_csv(data_path, low_memory=False)

# TODO 1: 找出哪种抱怨(Complaint Type)最多，将抱怨类型的名称提取出来
most_common_complaint = ...

# TODO 2: 看看哪个区(Borough)抱怨马路上噪声("Noise - Street/Sidewalk")的次数最多
noise_complaints = ...
noisiest_borough = ...

# TODO 3: 选出同时满足以下两个条件的记录：1.位于 Brooklyn 区; 2.抱怨类型包含 "Noise" 关键字
brooklyn_noise_mask = ...
brooklyn_noise_complaints = ...

# 不要修改下面的打印语句
print(f"Most common complaint: {most_common_complaint}")
print(f"Noisiest borough for Street/Sidewalk noise: {noisiest_borough}")
print(f"Total Brooklyn noise complaints: {len(brooklyn_noise_complaints)}")

# %% [markdown]
"""
### 可视化练习：COVID-19 疫情趋势

最后，结合一下我们刚刚学的 Pandas 方法以及之前学过的绘图。我们读取了一份包含全球各个国家/省份每日确诊 `COVID-19` 病例的数据表 `covid-19-cases.csv`。

在这个练习中，你需要：
1. 把各省份的数据按国家 `Country/Region` 聚合。
2. 找出截至最后一天，确诊人数最多的 **5 个国家**。
3. 利用 Matplotlib，将这 5 个国家各自的**总确诊趋势**和**每日新增确诊曲线**画出来。

"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/covid-19-cases.csv'
if not os.path.exists(data_path):
    data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/covid-19-cases.csv'
    
# 读取 COVID-19 确诊病例数据
cases = pd.read_csv(data_path)

# 数据的结构是：前四列为Province/State, Country/Region, Lat, Long，其后每一列是一个日期的确诊数

# TODO 1: 原数据包含不同省份(Province/State)，请将其按国家("Country/Region")进行分组并对各项数据求和
country_cases = ...

# TODO 2: 找到截至最后一天累积确诊人数最多的前 5 个国家
top5_countries = ...

# TODO 3: 画出这 5 个国家的累积确诊病例变化趋势和每日新增(增长量)曲线
# -------- 绘图部分 --------
# 提示：为了方便 Pandas 快速画图，建议你先清洗出纯时间序列数据，并将数据转置（使得日期成为索引，国家成为列）。
# top5_timeseries = ...

# fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# 1. 在 axes[0] 绘制累积确诊数量
# ...你的代码...

# 2. 计算每日新增病例，并在 axes[1] 绘制其曲线
# ...你的代码...

# plt.tight_layout()
# plt.show()
