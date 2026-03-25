# %% [markdown]
"""
# 多元线性回归 (Multiple Linear Regression)

## 1. 什么是多元线性回归？

多元线性回归是简单线性回归的扩展。它使用**两个或多个自变量**来预测一个因变量。

### 1.1 数学模型

假设我们有 $p$ 个自变量 $x_1, x_2, \dots, x_p$，多元线性回归的方程为：
$$ \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p $$

*   $\hat{y}$：预测的因变量。
*   $\beta_0$：截距 (Intercept)。
*   $\beta_1, \beta_2, \dots, \beta_p$：偏回归系数 (Partial Regression Coefficients)。$\beta_1$ 表示在**其他所有自变量保持不变**的情况下，$x_1$ 每增加 1 个单位，$y$ 平均增加的数量。

### 1.2 几何意义

*   **简单线性回归 (1 个自变量)**：在二维平面上拟合一条**直线**。
*   **二元线性回归 (2 个自变量)**：在三维空间中拟合一个**平面**。
*   **多元线性回归 (3 个及以上自变量)**：在高维空间中拟合一个**超平面 (Hyperplane)**（虽然我们无法画出来，但数学原理是一样的）。

![Multiple Linear Regression Plane](https://upload.wikimedia.org/wikipedia/commons/b/b0/Linear_least_squares_example2.svg)
*(图示：二元线性回归。红色的点是三维空间中的真实数据，蓝色的网格是我们拟合出的回归平面。)*

"""

# %% [markdown]
"""
## 2. 实例：预测信用卡年消费额

假设我们有一组真实的客户数据 (`credit_card_data.xlsx`)，记录了：
1.  **年收入 (Annual Income)**：$x_1$
2.  **高中毕业后受教育年数 (Years Post-High School Education)**：$x_2$
3.  **信用卡年消费额 (Annual Credit Card Charges)**：$y$

部分数据抽样展示如下： 



| 年收入 Annual Income ($x_1$) | 受教育年数 Years Education ($x_2$) | 消费额 Credit Charges ($y$) |
| :--- | :--- | :--- |
| $39,400 | 5 | $10,120.45 |
| $68,200 | 4 | $15,289.80 |
| $43,000 | 6 | $5,937.19 |
| $53,600 | 2 | $0.00 |
| $53,500 | 4 | $13,569.89 |
| $60,800 | 2 | $3,889.79 |
| $74,900 | 1 | $7,715.47 |

### 2.1 Pandas 数据读取复习与三维 EDA

*   **Pandas 读取复习**：`pd.read_excel('文件路径')` 是极为常见的数据加载方式。只要指定正确的路径，Pandas 就能将 Excel 表格无损转化为 `DataFrame` 数据框。如果你需要直接选取特定列用于计算，可以使用极其强大的 `iloc[:, :-1]` （提取除了最后一列的所有特征做 $X$）或 `iloc[:, -1]`（专门剥离提取最后一列作为 $y$）进行矩阵切割。
*   **三维可视化**：在探索性数据分析 (EDA) 阶段，由于我们面临的是 3 个维度的变量（两个 $X$，一个 $y$），普通的二维平面图已经不够用了。我们可以使用 matplotlib 的 `projection='3d'` 轻松画出立体散点图。

"""

# %%
import pandas as pd
from matplotlib import pyplot as plt
import os

# 1. 直接读取外部真实的 Excel 数据源文件
# 【注意路径】：这里做了环境兼容，无论您是在当前目录打开还是在根目录打开，都能顺利读取
file_path = '../../data/06_unit_regression/credit_card_data.xlsx'
if not os.path.exists(file_path):
    file_path = '../../data/06_unit_regression/credit_card_data.xlsx'

df = pd.read_excel(file_path)

# 2. 3D 三维图展示
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')  # 激活三维画板引擎
# 分别取第 0，1，2 列作为 X, Y, Z 轴
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], color='navy', s=40)
ax.set_xlabel('Income')
ax.set_ylabel('Education')
ax.set_zlabel('Charges')
ax.set_title('3D Scatter of Income, Education and Charges')
plt.show()




# %% [markdown]
"""
### 2.2 散点图矩阵与相关性深度挖掘

`sns.pairplot(df)` （以及 pandas 自带的 `scatter_matrix`） 这个函数可以把表格里所有的特征进行“强行相亲”：将所有变量两两配对画在散点图中，其对角线则退化为单变量的分布直方图。

**核心诊断原则：**
1.  **理想模型**：你希望自变量 ($x_1, x_2$) 之间最好没有任何关系（彼此独立、各司其职），但它们都分别跟因变量 ($y$) 存在高度关联倾向。
2.  **共线性警告**：如果自变量之间发生了强烈相互纠缠关联，这叫“多重共线性”。我们以后必须考虑删掉其中一个废弃关联件。

![Scatter Matrix Insight](images/2020-02-14-01-20-01.png)

### 2.3 多变量高维多项式拟合

之前的多项式拟合只针对单一 $x$，而面对多个 $X$，`PolynomialFeatures` 依旧能够通吃：它会将 $X_1$ 与 $X_2$ 产生 $[1, x_1, x_2, x_1^2, x_2^2, x_1 x_2 \dots]$ 这种恐怖数量级的升阶阵列，并继续借助 `sm.OLS` 完成统治。

"""

# %%
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

# 散点图矩阵排查替代方案（解决旧版本 pyplot 与 pandas 不兼容导致的 AttributeError 报错）：
# 直接使用 Seaborn 的 pairplot 平替 pd.plotting.scatter_matrix
sns.pairplot(df, diag_kind='hist', plot_kws={'color': 'green', 'alpha': 0.6})
plt.suptitle('Pairwise Scatter Matrix Check', y=1.02)
plt.show()

# 打印相关系数矩阵
print("--- 相互相关系数体检 ---")
print(df.corr())

# 提取特征列和标签列，X现在是两列(不用reshape)，y 是末尾一列(需强行拉成二维)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# PolynomialFeatures 引擎直接对多个 X 联手进行高阶魔改升维
# 这里直接给到 3 次方，由于多组合效应，维度爆炸会非常大
poly_reg = PolynomialFeatures(degree=3)
poly_X = poly_reg.fit_transform(X)

# 老规矩，statsmodels 大爷不伺候自动截距，必须人肉手动补全
poly_X_int = sm.add_constant(poly_X)

# 喂入硬核检验的 OLS 高炉进行计算
poly_model = sm.OLS(y, poly_X_int)
poly_res = poly_model.fit()

# 查看这狂暴三阶模型极其细致的总结审判书
print(poly_res.summary())




# %% [markdown]
"""
## 3. 回归结果解读与模型评估

面对上方海量的 `summary` 报告信息输出，对于多元回归主要盯紧以下核心生死指标：

1.  **Adj. R-squared (调整后多元决定系数)**：
    *   在包含多变量 $X$ 时，普通的 $R^2$ 会出现**注水造假**（哪怕增加完全无关的垃圾特征，$R^2$ 依然会扩大）。
    *   `Adj. R-squared` 内部内置了“过多无效自变量惩罚机制”，它是评估此类复杂多自变量组团回归时最严谨的参考标杆。
2.  **P>|t| 单变量生死 P 值检验**：
    *   用以裁判具体每一项特定的 $x_i$ 自变量对整体 $y$ 的贡献是否有真实的显著作用。
    *   **法则：若 P < 0.05，此变量保留；若 P > 0.05，则可以直接判定该特征为废柴无用干扰项。** 如果在多变量中发现某特征不显著，极可能说明它跟其他变量存在了“功能重叠”的共线性。

### 3.1 补充诊断：残差校验
和简单回归一样，合格的多自变量模型跑出后，其散兵游勇残点（`resid`）绝不允许呈现漏斗型、U型等非随机的异形。

"""

# %%
import matplotlib.pyplot as plt

# 分别针对不同的单一变量横轴视角，把诊断残差散点靶心图投射出来
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], poly_res.resid, color='purple', alpha=0.7)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('Residuals targeting $x_1$ Income')
axes[0].set_xlabel('Income')

axes[1].scatter(X[:, 1], poly_res.resid, color='orange', alpha=0.7)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('Residuals targeting $x_2$ Education')
axes[1].set_xlabel('Education')

plt.tight_layout()
plt.show()




# %% [markdown]
"""
### 动手综合试炼：医学全流线预处理与风险预测建模

当你面对一组崭新的医院体检数据集 `stroke_risk_data.xlsx`：旨在根据**年龄 (Age $x_1$)**、**血压 (Blood Pressure $x_2$)** 和 **吸烟史 (Smoker $x_3$)** 来推断预测 **未来中风瘫痪的风险率 (% Risk $Y$)**。

| Age ($x_1$) | Blood Pressure ($x_2$) | Smoker ($x_3$) | % Risk of Stroke ($Y$) |
| :--- | :--- | :--- | :--- |
| 63 | 129 | No | 7 |
| 75 | 99  | No | 15 |
| 80 | 121 | No | 31 |
| 82 | 125 | No | 17 |
| 60 | 134 | No | 14 |
| 79 | 205 | Yes | 48 |
| 79 | 120 | Yes | 36 |

**核心痛点发现**：
机器无法读懂人类的主观单词。模型遇到第三列 "Smoker" 里的 'Yes' 和 'No' 直接就会由于数学计算错误崩盘死机。

**实战终极打通任务**：
1. **清洗预处理**：利用 Pandas 的 `replace` 大法，将表格内的 `Yes` 擦除成 `1`，`No` 转为 `0` （也就是所谓的类别二值编码）。
2. **矩阵切割**：准确地将包含 $x_1, x_2, x_3$ 的前三列提为自变量 $X$，讲末尾列提为受验 $Y$。
3. **推演建模**：调动 `sm.OLS` 搭建多元回归评估模型，强制补充常数截距列后拟合出数据结果。
4. **提交报告**：将这批医学数据的 `summary()` 报告重磅打印公开出来查阅其特征生存生死状态！

"""

# %%
# --------- 【在此完成预处理并跑通 OLS 模型】 ---------
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# 环境路径兼容加载
quiz_file = '../../data/06_unit_regression/stroke_risk_data.xlsx'
if not os.path.exists(quiz_file):
    quiz_file = '../../data/06_unit_regression/stroke_risk_data.xlsx'
    
df_quiz = pd.read_excel(quiz_file)

# 任务 1：请在下方使用 replace 函数将 'Yes' 替换为 1，'No' 替换为 0 (需配置 inplace=True)
# --------------------------------------------------



# 任务 2：提取特征矩阵 X (前三列) 与标签 Y (最后一列)
# --------------------------------------------------




# 任务 3：使用 statsmodels (sm) 构建并拟合 OLS 多元回归模型
# 提示：别忘了模型大忌，要手动使用 sm.add_constant 为 X 补充截距项！
# --------------------------------------------------




# 任务 4：打印这份医学报告的 summary()
