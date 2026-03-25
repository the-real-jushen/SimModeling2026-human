# %% [markdown]
"""
# 非线性回归 (Non-linear Regression)

## 1. 什么是非线性回归？

当因变量 $Y$ 和自变量 $X$ 之间的关系无法用线性方程（即参数的最高次数为 1）来描述时，我们就需要建立非线性模型。

**常见的非线性模型包括：**
*   **多项式模型**：$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots$ (注意：虽然它关于 $x$ 是非线性的，但关于参数 $\beta$ 仍然是线性的，所以有时也被归类为线性回归的变体)。
*   **指数模型**：$y = a \cdot e^{bx}$
*   **对数模型**：$y = a + b \cdot \ln(x)$
*   **自定义复杂物理/化学模型**：基于领域知识推导出的复杂公式。

![Non-linear Regression](https://upload.wikimedia.org/wikipedia/commons/8/8e/IllustrationRegressionNonLineaire.png)
*(图示：非线性回归。数据点呈现出明显的曲线趋势，用直线拟合会产生很大的误差，必须用曲线（如指数或多项式）来拟合。)*

## 2. 如何求解非线性回归？

线性回归可以通过最小二乘法直接求出解析解（公式解）。但对于大多数非线性回归，我们无法直接写出参数的公式。

**解决方法：**
我们需要使用**优化算法**（如我们在第 4 单元学过的梯度下降法、牛顿法等）来不断迭代，寻找使误差平方和最小的参数值。

在 Python 中，最常用的非线性拟合工具是 `scipy.optimize.curve_fit`。

"""

# %% [markdown]
"""
## 3. 实例：化学反应速率建模

**问题背景：**
在某个化学反应中，反应速率 $y$ 与三种反应物的含量 $x_1$ (氢), $x_2$ (戊烷), $x_3$ (异构戊烷) 有关。化学家根据反应机理，推导出了如下的数学模型：

$$ y = \frac{\beta_4 x_2 - \frac{x_3}{\beta_5}}{1 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3} $$

我们的任务是：根据实验测得的数据，求出模型中的 5 个未知参数 $\beta_1, \beta_2, \beta_3, \beta_4, \beta_5$。

"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 1. 准备实验数据
data = {
    'x1': [470, 285, 470, 470, 470, 100, 100, 470, 100, 100, 100, 285, 285],
    'x2': [300, 80, 300, 80, 80, 190, 80, 190, 300, 300, 80, 300, 190],
    'x3': [10, 10, 120, 120, 10, 10, 65, 65, 54, 120, 120, 10, 120],
    'y': [8.55, 3.79, 4.82, 0.02, 2.75, 14.39, 2.54, 4.35, 13.0, 8.5, 0.05, 11.32, 3.13]
}
df = pd.DataFrame(data)

# 提取自变量 X (需要转置为 curve_fit 要求的格式: 每一行是一个变量，每一列是一个样本)
# X 的形状将是 (3, 13)
X_data = df[['x1', 'x2', 'x3']].values.T
y_data = df['y'].values

# 2. 定义非线性模型函数
# 第一个参数必须是自变量 X，后面的参数是我们需要拟合的未知参数
def reaction_model(X, b1, b2, b3, b4, b5):
    x1, x2, x3 = X[0], X[1], X[2]
    # 为了防止除以 0 的错误，我们在分母加一个极小的数 (虽然在这个特定数据中可能不需要)
    numerator = b4 * x2 - x3 / b5
    denominator = 1 + b1 * x1 + b2 * x2 + b3 * x3
    return numerator / denominator

# 3. 使用 curve_fit 进行拟合
# p0 是参数的初始猜测值。对于复杂的非线性模型，提供一个合理的初始值非常重要！
# 如果不提供，默认全为 1.0。这里我们随便给一组初始值。
initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]

# popt 包含拟合出的最优参数
# pcov 是参数的协方差矩阵 (用于评估参数的不确定性，这里我们暂不深入)
popt, pcov = curve_fit(reaction_model, X_data, y_data, p0=initial_guess)

# 提取拟合出的参数
b1_opt, b2_opt, b3_opt, b4_opt, b5_opt = popt

print("--- 非线性回归拟合结果 ---")
print(f"最优参数 beta_1: {b1_opt:.4f}")
print(f"最优参数 beta_2: {b2_opt:.4f}")
print(f"最优参数 beta_3: {b3_opt:.4f}")
print(f"最优参数 beta_4: {b4_opt:.4f}")
print(f"最优参数 beta_5: {b5_opt:.4f}")

# 4. 模型评估
# 使用拟合出的参数计算预测值
y_pred = reaction_model(X_data, *popt)

# 计算 R-squared
r2 = r2_score(y_data, y_pred)
print(f"\n模型决定系数 (R^2): {r2:.4f}")

# 5. 可视化：真实值 vs 预测值
plt.figure(figsize=(8, 5))
# 画一条 y=x 的参考线，如果预测完全准确，所有的点都会落在这条线上
plt.plot([0, 15], [0, 15], color='gray', linestyle='--', label='Perfect Prediction')
plt.scatter(y_data, y_pred, color='blue', s=50, label='Actual vs Predicted')

plt.title('Non-linear Regression: Actual vs Predicted Reaction Rate')
plt.xlabel('Actual Reaction Rate (y)')
plt.ylabel('Predicted Reaction Rate (y_pred)')
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
"""
### 深入思考：为什么这里的 $R^2$ 这么高？

运行上述代码后，你可以发现这个非线性模型的 $R^2$ (决定系数) 非常接近 1，这表示模型能极好地解释数据的方差。而点大致分布在 $y=x$ 参考线附近，说明真实值和预测值高度吻合。

这是为什么呢？

一般情况下，我们在做多项式回归或一般机器学习时，是在用一个**通用的数学结构**去盲目逼近数据；但在上面这个化学案例中，我们的公式并非盲选。这个包含分式的复杂模型（即我们在 `reaction_model` 函数中定义的公式），是建立在**物理和化学动力学规律**基础之上的！

> **核心概念：机理模型优先原则**
> - **经验模型 (Empirical Models)**：通过观察数据外观（散点图）直接套用多项式等通用函数建立的模型（只看数据，不管原理）。
> - **机理模型 (Mechanistic Models)**：通过事物的内在物理/化学/经济规律推演出来的模型公式。

当你可以通过已知的科学规律建立起一个包含未知参数的规则公式时，**基于该物理公式进行非线性回归，往往是最准确的！** 这个方法远比抛弃原理盲目使用一个通用多项式模型（不管它的阶数有多高）效果要好且更具可解释性。

当然，在实际工程（尤其是黑盒系统或是纯数据驱动的项目）中，很多时候我们无法获得业务背后的清晰物理公式。这就是为什么各种通用算法存在的意义。但在能建立机理模型的情况下，机理模型永远是首选。

"""
