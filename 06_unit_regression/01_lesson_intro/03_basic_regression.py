# %% [markdown]
"""
# 回归的基本概念与简单线性回归

## 1. 什么是回归？

**回归 (Regression)** 是一种统计学方法，用于寻找一个或多个自变量 (Independent Variables) 与一个因变量 (Dependent Variable) 之间的关系。

*   **自变量 ($X$)**：我们已知的、用来做预测的变量（也叫特征 Feature）。
*   **因变量 ($Y$)**：我们想要预测的、未知的变量（也叫标签 Label 或目标 Target）。

**回归的特点：**
1.  **预测连续值**：回归模型的输出结果只能是一个连续的变量不论是 ratio（等比）还是 interval（等距）。如果预测出的结果是有限离散的（如“下雨”或“晴天”），这可能就是分类或者聚类问题了。
2.  **发现关系**：回归不仅能直接给出预测值，还能告诉你变量之间的关系。你可以看出，当某个自变量增加时，因变量是增加还是减少，影响有多大。
3.  **相关性不等于因果关系 (Causality)**：回归模型发现的是规律，绝不代表因果。例如，通过分析我们可能发现7-10岁的小孩中“脚越大，智商越高！”。脚和脑子有啥关系？更加理性的分析会指出：由于脚越大的小孩往往年龄也越大，所以智商随之变高。运用机器学习方法时必须时刻注意这些问题。

## 2. 简单线性回归 (Simple Linear Regression)

最简单的回归模型就是**简单线性回归**。它假设自变量 $X$ 和因变量 $Y$ 之间存在一条直线的关系。

### 2.1 数学基本概念

简单线性回归的公式非常简单，就是我们在中学学过的直线方程：
$$ \hat{y} = \beta_0 + \beta_1 x $$

*   $\hat{y}$ (读作 y-hat)：模型**预测**的因变量的值。这本质上表示在给定 $x$ 的情况下 $y$ 的**期望值**，即 $\hat{y} = E(Y|X)$。
*   $x$：自变量 (Independent Variable) 的值。
*   $\beta$：只有确定了模型参数 $\beta_0, \beta_1$ 才是一个真正的模型。$\beta_0$ 是截距，$\beta_1$ 是斜率/权重。

**真实观测值 (Ground Truth) 与误差：**
在现实中，数据点一般不会完美落在一条直线上。真实的观测数据 $y$ 实际上是在模型基础上增加了一个误差的：
$$ y = \beta_0 + \beta_1 x + \epsilon $$
其中 $\epsilon$ (Epsilon) 代表观测和系统本身存在的噪声导致的**误差**。

**关键概念防坑：到底什么是“线性”回归？**
你是否一直以为线性回归的特征只能是 $x$ 的一阶等比直关系？错！**线性回归是一个回归模型，它是关于“参数 $\beta$ ”的线性函数**。而不是要求自变量 $x$ 也是这种线性函数结构。
因此，下面这些所有的模型，**全部都属于线性回归**模型：
*   $\hat{y} = \beta_0 + \beta_1 x$
*   $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2^2 + \beta_3 x_3^3$
*   $\hat{y} = \beta_0 + \beta_1 \sin(x)$
不管你的 $x$ 在模型函数中是不是线性，假设你有 $\sin(x)$，把整个 $\sin(x)$ 作为一个新的黑盒自变量扔进模型中就可以了，因此对于参数 $\beta$ 它们永远是一阶线性的。

![Linear Regression Residuals](https://upload.wikimedia.org/wikipedia/commons/e/ec/Linear_regression_residuals.svg)
*(图示：蓝色的点是真实的观测数据，红色的线是我们拟合的回归模型。红线和蓝点之间的垂直虚线就是误差/残差。)*

"""

# %% [markdown]
"""
## 3. 普通最小二乘法 (Ordinary Least Squares, OLS)

**目标：** 找到一组参数 ($\beta_0$ 和 $\beta_1$)，使得模型预测的直线尽可能地贴近所有的真实数据点。

**方法：** 我们希望所有的误差 $\epsilon_i = y_i - \hat{y}_i$ 越小越好。为了避免正负误差相互抵消，我们计算误差的**平方和**。

这个误差平方和被称为 **均方误差 (Mean Squared Error, MSE)** 或 **残差平方和 (Sum of Squared Residuals, SSR)**：
$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2 $$

**OLS 的核心思想就是：通过数学推导（求导求极值），找到能让 MSE 达到最小值的 $\beta_0$ 和 $\beta_1$。**

*OLS的隐藏假设：*
使用这种方法有一个非常关键的假设需要你记住：我们的这些误差 $\epsilon$ 纯粹是由于观测本身和 $y$ 数据固有造成的噪声。也就是说我们在拿出一对样本数据 $(x_i, y_i)$ 进行预测时候，**$x_i$ 的数据特征观测被要求是绝对准确没有任何误差的**。

在 Python 中，我们不需要手动去求导，强大的机器学习库 `scikit-learn` (简称 `sklearn`) 已经帮我们封装好了这一切。

"""

# %% [markdown]
"""
### 知识小测验：究竟什么是“线性”回归？

为了确保你真的懂了线性回归的精髓定义，请回答以下选项中正确的是？


**选项:**
- A. 只有 y = beta_0 + beta_1 * x 属于线性回归
- B. y = beta_0 + beta_1 * x^2 + beta_2 * x^3 不属于线性回归
- C. y = beta_0 + beta_1 * sin(x) 不属于线性回归
- D. 以上全部式子其实都属于且可以是线性回归模型

**正确答案:** D. 以上全部式子其实都属于且可以是线性回归模型
**提示:** 仔细回忆前面关于线性回归定义的防坑说明，线性回归不限制自变量 x 的分布状态，而是看 '参数 beta' 是否呈现一阶线性关系！

"""

# %% [markdown]
"""
## 4. 代码实战：身高与腿长的线性回归

> **实战任务描述**：测得 16 名成年女子的身高与腿长，所得数据如下表所示。我们希望探索“身高”与“腿长”之间是否存在线性关系。请以身高 $x$ 为自变量（横坐标），以腿长 $y$ 为因变量（纵坐标），利用普通最小二乘法 (OLS) 构建一个简单线性回归模型，并用该模型来预测身高为 165cm 的女子的腿长。

| 身高 (cm) | 143 | 145 | 146 | 147 | 149 | 150 | 153 | 154 | 155 | 156 | 157 | 158 | 159 | 160 | 162 | 164 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **腿长 (cm)** | **88** | **85** | **88** | **91** | **92** | **93** | **93** | **95** | **96** | **98** | **97** | **96** | **98** | **99** | **100** | **102** |

在接下来的代码中，我们将使用 Python 中堪称行业标准的机器学习库 `scikit-learn` 来训练模型并完成预测。

"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 准备数据
# 身高 (x) 和 腿长 (y)
heights = np.array([143, 145, 146, 147, 149, 150, 153, 154, 155, 156, 157, 158, 159, 160, 162, 164])
leg_lengths = np.array([88, 85, 88, 91, 92, 93, 93, 95, 96, 98, 97, 96, 98, 99, 100, 102])

# sklearn 要求输入的特征 X 必须是二维数组 (矩阵)，形状为 (样本数, 特征数)
# 我们的特征只有一个 (身高)，所以需要将一维数组 reshape 为 (16, 1)
X = heights.reshape(-1, 1)
y = leg_lengths

# 2. 创建并训练模型
# 初始化线性回归模型
model = LinearRegression()

# 使用 fit() 方法训练模型 (这就是 OLS 寻找最佳参数的过程)
model.fit(X, y)

# 3. 获取模型参数
beta_0 = model.intercept_  # 截距
beta_1 = model.coef_[0]    # 斜率

print(f"模型训练完成！")
print(f"截距 (beta_0): {beta_0:.4f}")
print(f"斜率 (beta_1): {beta_1:.4f}")
print(f"回归方程: 腿长 = {beta_0:.4f} + {beta_1:.4f} * 身高")

# 4. 使用模型进行预测
# 假设有一个身高为 165cm 的女生，预测她的腿长
new_height = np.array([[165]])
predicted_leg_length = model.predict(new_height)
print(f"\n预测身高 165cm 的腿长: {predicted_leg_length[0]:.2f} cm")

# 计算所有训练数据的预测值
y_pred = model.predict(X)

# 5. 模型评估
# MSE (均方误差): 越小越好
mse = mean_squared_error(y, y_pred)
# R-squared (决定系数): 衡量模型拟合优度的指标，取值范围 0 到 1，越接近 1 说明模型解释力越强
r2 = r2_score(y, y_pred)

print(f"\n模型评估指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R^2): {r2:.4f}")

# 6. 可视化结果
plt.figure(figsize=(8, 5))
# 画出真实的散点数据
plt.scatter(heights, leg_lengths, color='blue', label='Actual Data')
# 画出模型拟合的直线
plt.plot(heights, y_pred, color='red', linewidth=2, label='Regression Line')

plt.title('Simple Linear Regression: Height vs Leg Length')
plt.xlabel('Height (cm)')
plt.ylabel('Leg Length (cm)')
plt.legend()
plt.grid(True)
plt.show()



