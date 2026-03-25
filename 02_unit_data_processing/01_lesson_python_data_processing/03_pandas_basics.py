# %% [markdown]
"""
# Pandas 数据处理基础

在科学计算和数据分析中，**Pandas** 是不可或缺的工具。如果说 NumPy 是 Python 的矩阵计算器，那么 Pandas 就是 Python 的 Excel。

Pandas 的核心数据结构是 **DataFrame**。你可以把它想象成一个带有行标签（Index）和列标签（Columns）的二维表格。

![Pandas DataFrame](https://upload.wikimedia.org/wikipedia/commons/9/9c/Pandas_dataframe.png)
*(图示：Pandas DataFrame 的结构，包含行索引、列名和数据)*

**本节课你将学到：**

1.  如何使用 Pandas 读取 CSV 文件。
2.  如何查看数据的基本信息和统计摘要。
3.  如何使用 `loc` 和 `iloc` 选择特定的行、列或单元格。
4.  如何进行简单的数据清洗（处理缺失值和重复值）。

"""

# %% [markdown]
"""
## 1. 读取数据与基本信息

为了演示，我们假设有一个名为 `data.csv` 的文件（在实际运行中，如果文件不存在会报错，这里我们先用代码生成一个简单的假数据文件供后续读取）。

"""

# %%
import pandas as pd
import numpy as np

# --- 准备工作：生成一个假的 data.csv 文件供演示使用 ---
# (在实际应用中，你通常直接读取已有的文件)
# 我们已经将真实数据放在了 data 目录下，所以这里不再需要生成假数据了。
# ---------------------------------------------------------

# 1. 读取 CSV 文件
# 默认情况下，Pandas 会自动将第一行作为列名 (Columns)，并自动生成从 0 开始的数字索引 (Index)
# 注意：这里我们使用相对路径读取数据文件
# 假设当前的运行目录是该脚本所在的目录，向上一级为 `02_unit_data_processing`，再向上一级到了 `course`，然后进入 `data/02_unit_data_processing/...`
data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/health_data.csv'
import os
if not os.path.exists(data_path):
    # Fallback to workspace root relative access if running from workspace root
    data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/health_data.csv'

df = pd.read_csv(data_path)

# 2. 查看前 5 行数据
print("--- df.head() ---")
print(df.head())

# 3. 查看数据的基本信息 (行数、列数、每列的数据类型、非空值数量)
print("\n--- df.info() ---")
df.info()

# 4. 查看数值列的统计摘要 (均值、标准差、最小值、最大值、四分位数)
print("\n--- df.describe() ---")
print(df.describe())




# %% [markdown]
"""
## 2. 选择与过滤数据

### `loc`：基于标签 (Label-based) 选择
`loc` 使用行索引的名字和列的名字来选择数据。
语法：`df.loc[行标签, 列标签]`

"""

# %%
# 1. 选择特定的一行 (返回一个 Series)
print("--- 选择索引为 2 的行 ---")
print(df.loc[2])

# 2. 选择多行 (注意：loc 的切片是包含两端的！)
print("\n--- 选择索引 1 到 3 的行 ---")
print(df.loc[1:3])

# 3. 选择特定的行和列
print("\n--- 选择索引为 5 的行的 'Pulse' 和 'Calories' 列 ---")
print(df.loc[5, ["Pulse", "Calories"]])




# %% [markdown]
"""
### `iloc`：基于位置 (Integer position-based) 选择
`iloc` 使用行和列的整数坐标来选择数据。
语法：`df.iloc[行位置, 列位置]`

"""

# %%
# 1. 选择第 3 行 (索引为 2)
print("--- 选择第 3 行 ---")
print(df.iloc[2])

# 2. 选择第 2 到第 4 行 (注意：iloc 的切片是左闭右开的，不包含结束位置，这与 Python 列表一致)
print("\n--- 选择第 2 到第 4 行 (位置 1:4) ---")
print(df.iloc[1:4])

# 3. 选择特定的行和列 (例如：第 3 到 5 行，第 1 和第 2 列)
print("\n--- 选择特定位置的行和列 ---")
print(df.iloc[2:5, [0, 1]])




# %% [markdown]
"""
### 条件过滤 (Boolean Indexing)

我们可以直接选择某一列，或者根据条件筛选出满足要求的行。

"""

# %%
# 1. 直接选择某一列 (返回一个 Series)
pulses = df['Pulse']
print("--- 'Pulse' 列的前 3 个值 ---")
print(pulses.head(3))

# 2. 条件过滤：找出所有 Pulse > 110 的行
# df['Pulse'] > 110 会返回一个由 True 和 False 组成的 Series
# 将这个 Series 传给 df[]，就会只保留 True 对应的行
high_pulse_df = df[df['Pulse'] > 110]

print("\n--- 心率大于 110 的记录 ---")
print(high_pulse_df)

# 3. 组合条件：Pulse > 110 且 make 是 'honda'
# 注意：多个条件必须用括号括起来，并使用 & (与) 或 | (或) 连接
complex_filter = df[(df['Pulse'] > 110) & (df['make'] == 'honda')]
print("\n--- 心率大于 110 且品牌为 honda 的记录 ---")
print(complex_filter)




# %% [markdown]
"""
## 3. 其他常用数据操作

在实际处理数据时，我们还需要不时改变数据的形态或进行基本的值统计。

"""

# %%
# 1. 转换为 Numpy 数组
df_array = df.head(3).to_numpy()
print("--- 转换为 Numpy 数组 (前 3 行) ---")
print(df_array)

# 2. DataFrame 转置 (行列互换)
print("\n--- DataFrame 转置 (前 3 行的转置) ---")
print(df.head(3).T)

# 3. 设置某列为索引 
df_indexed = df.set_index("make")
print("\n--- 将 'make' 列设置为索引后，使用 loc 查找 'honda' ---")
# 注意：这可能会返回多行
if "honda" in df_indexed.index:
    print(df_indexed.loc["honda"].head(2))

# 4. 统计某列中不同值的出现次数
print("\n--- 统计 'make' 列中各项出现的频率 ---")
print(df['make'].value_counts())

# 5. 获取某列所有的唯一值
print("\n--- 'make' 列的唯一值 ---")
print(df['make'].unique())

# 6. 新增行与列，并删除列
df_copy = df.copy()
# 增加一行 (如果 index 为新，相当于追加)
new_index = df_copy.index.max() + 1
df_copy.loc[new_index] = [60, 120, 140, 300.0, "mazda"] # 给定一个与列数匹配的列表

# 增加一列：通过计算其他列得到
df_copy["Pulse_Double"] = df_copy["Pulse"] * 2

# 或者用 insert 指定插入位置
# df_copy.insert(1, "Pulse_Double", df_copy["Pulse"] * 2, True)

# 删除刚才那列
df_copy.drop(columns=["Pulse_Double"], inplace=True)
print("\n--- 增删行列操作执行完毕 ---")




# %% [markdown]
"""
### 知识检查

如果我想使用**基于位置**的方法（忽略可能存在的字符串行标签），严格提取 DataFrame 的**前 3 行**，以下哪行代码是正确的？


**选项:**
- df.loc[0:2]
- df.iloc[0:2]
- df.loc[0:3]
- df.iloc[0:3]

**正确答案:** df.iloc[0:3]
**提示:** `iloc` 是基于位置的，且切片规则与 Python 列表相同（左闭右开）。要获取前 3 行（位置 0, 1, 2），切片应该是 `0:3`。

"""
