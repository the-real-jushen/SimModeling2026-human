# %% [markdown]
"""
# 数据拟合 (Curve Fitting)

拟合是数学建模中最核心的步骤之一。当我们观察到一组数据，并猜测它们之间存在某种数学关系（如线性、指数、多项式）时，我们需要通过拟合来确定这个数学模型中的未知参数。

**本节课你将学到：**

1.  理解插值与拟合的区别。
2.  使用 NumPy 的 `polyfit` 进行简单的多项式拟合。
3.  掌握最小二乘法的数学核心思想。
4.  使用 SciPy 的 `curve_fit` 进行任意自定义函数的拟合。

"""

# %% [markdown]
"""
## 1. 原理：最小二乘法 (Least Squares)

在运行代码前，我们必须理解拟合背后的数学原理：**寻找一组误差最小的参数**。

假设我们有一组带误差的观测点 $(x_i, y_i)$，并且猜测其遵循某种真实模型规律 $f(x, eta)$（例如直线模型 $f(x) = eta_0 + eta_1 x$）。
模型在你给定的每一个测量点 $x_i$ 处，都会给出一个理论预测值 $\hat{y}_i = f(x_i, eta)$。
此时，实验真实值 $y_i$ 与理论预测值 $\hat{y}_i$ 之间就产生了**残差（Residual）**：$e_i = y_i - \hat{y}_i$。

最小二乘法的核心思想就是：**寻找一组最佳参数组合 $eta$，使得所有数据点的残差平方和（Sum of Squared Residuals, SSR）最小化**：
$$ SSR(eta) = \sum_{i=1}^{n} (y_i - f(x_i, eta))^2 $$

**为什么必须要用“平方”再求和？**
1. 防止相互抵消：如果不用平方，正误差和负误差加起来可能会变成 0，让你误以为拟合得很完美。
2. 惩罚大误差：平方操作会把巨大的偏差放大，迫使拟合曲线在整体上对所有离群点更具有均衡的照顾性。
3. 导数友好：平方函数是个平滑的抛物面，求解最小值极为方便（对其求导等于 0 即可锁定最优解极值点）。

几乎所有 NumPy 和 SciPy 的底层拟合方法，本质上都在执行这套求极小值参数的运算法则。

"""

# %% [markdown]
"""
## 2. 多项式拟合 (Polynomial Fitting)

**为什么要用多项式拟合？**
在工程初期，我们往往不知道数据背后隐藏的最佳物理公式是什么。根据**泰勒展开定理（Taylor's Theorem）**，任何平滑的连续函数在局部都可以被任意精度的多项式 $P(x) = a_0 + a_1x + a_2x^2 + \dots + a_nx^n$ 足够逼近。因此，多项式拟合是一种非常强大的“万金油”盲测手段，特别适合在缺乏理论物理模型支撑时，对数据趋势进行探索性估算。

对于简单的多项式回归（无论是直线 $y=ax+b$ 还是抛物线 $y=ax^2+bx+c$），由于其各项参数呈线性分布，底层的最小二乘法可以直接化简成为简单的矩阵乘法求解。NumPy 提供了非常轻量的 `polyfit` 函数，它利用奇异值分解（SVD）算法可以瞬间计算出最优解系数组。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成带有噪声的模拟数据
# 假设真实模型是 y = 0.5 * x^2 - 2 * x + 3
np.random.seed(42) # 设置随机种子以保证结果可重复
x_data = np.linspace(-5, 5, 30)
y_true = 0.5 * x_data**2 - 2 * x_data + 3
# 添加正态分布的噪声
y_data = y_true + np.random.normal(0, 2, size=x_data.shape)

# 2. 使用 np.polyfit 进行多项式拟合
# 参数：x数据, y数据, 多项式阶数(degree)
# 拟合一阶多项式 (直线)
coeffs_1 = np.polyfit(x_data, y_data, 1)
# 拟合二阶多项式 (抛物线)
coeffs_2 = np.polyfit(x_data, y_data, 2)

print("一阶拟合系数 (a, b):", coeffs_1)
print("二阶拟合系数 (a, b, c):", coeffs_2)

# 3. 使用 np.poly1d 构建多项式函数对象，方便计算预测值
p1 = np.poly1d(coeffs_1)
p2 = np.poly1d(coeffs_2)

# 4. 绘图对比
x_plot = np.linspace(-6, 6, 100) # 生成更密集的 x 用于画平滑曲线

plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='black', label='Noisy Data')
plt.plot(x_plot, p1(x_plot), 'r--', label='Degree 1 Fit (Linear)')
plt.plot(x_plot, p2(x_plot), 'b-', linewidth=2, label='Degree 2 Fit (Quadratic)')

plt.title("Polynomial Fitting using np.polyfit")
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
"""
## 3. 任意函数拟合 (Custom Function Fitting)

**为什么要用自定义函数，多项式不够用吗？**
多项式拟合虽然通用，但也存在致命弱点：不包含任何“物理意义”。多项式外推时往往会趋向无穷大（龙格现象），无法准确刻画自然界中普遍存在的边界收敛行为：例如电容器充电的**指数饱和**、弹簧振子的**正弦衰减**或人口繁殖的**S型逻辑斯蒂增长**。当你手里已经握有明确的物理/化学方程式时，生硬地去套用多项式是极度不精确的。

此时我们必须依赖 SciPy 提供的 `optimize.curve_fit`。

**`curve_fit` 的工作原理：**
多项式的参数在数学上属于“线性关系”（即使含有 $x^2$，但是它前面的系数 $a$ 依然是一次方的），所以一步矩阵运算就能得到确解。
但在非线性函数（例如 $\cos(C \cdot x)$ 中的参数 $C$ 被包裹在三角函数内）中，最小二乘法无法直接写出解析解！`curve_fit` 在底层调用了强大的**非线性最小二乘迭代算法（通常为莱文贝格-马夸特方法 Levenberg-Marquardt）**。它会：
1. 聪明的从一个初始猜测参数出发。
2. 内部计算偏导数/雅可比矩阵，判断梯度“下坡”最快的方向。
3. 动态更新参数，不断“试错”，直到残差平方和（SSR）陷入谷底不可降低为止，最后将最佳极值点参数输出。

**使用步骤：**
1.  **构建数学模型 (Objective Function)**：函数参数首位必须是接收阵列数据的自变量 `x`，后续参数需预留出待求的未知参数（如 `a, b, c`）。
2.  **调用 `curve_fit` 激活迭代**：将刚刚封好的模型与原始观测 $(x, y)$ 传入引掣，等待几毫秒的迭代下降算法冲向底峰。
3.  **提取结果**：它会返回最佳拟合参数元组（`popt`），和一个表示各参数间波动关系的协方差矩阵（`pcov`）。

"""

# %%
from scipy.optimize import curve_fit

# 1. 生成模拟数据：假设这是一个阻尼振荡信号
# 真实模型：y = A * exp(-B * x) * cos(C * x)
x_data2 = np.linspace(0, 4, 50)
y_true2 = 2.5 * np.exp(-1.3 * x_data2) * np.cos(1.5 * np.pi * x_data2)
y_data2 = y_true2 + np.random.normal(0, 0.2, size=x_data2.shape)

# 2. 定义我们要拟合的目标函数
# 注意：第一个参数必须是 x，后面跟着需要求解的参数 a, b, c
def damped_oscillator(x, a, b, c):
    return a * np.exp(-b * x) * np.cos(c * x)

# 3. 执行拟合
# popt (optimal parameters): 最佳参数数组 [a, b, c]
# pcov (covariance): 协方差矩阵
popt, pcov = curve_fit(damped_oscillator, x_data2, y_data2)

# 解包参数
a_fit, b_fit, c_fit = popt
print(f"拟合得到的参数: A={a_fit:.3f}, B={b_fit:.3f}, C={c_fit:.3f}")
print(f"真实的参数为: A=2.5, B=1.3, C={1.5*np.pi:.3f}")

# 4. 绘图
x_plot2 = np.linspace(0, 4.5, 200)
y_fit2 = damped_oscillator(x_plot2, a_fit, b_fit, c_fit)

plt.figure(figsize=(8, 5))
plt.scatter(x_data2, y_data2, label='Data')
plt.plot(x_plot2, y_fit2, 'r-', linewidth=2, label='Fitted Curve')
plt.title("Custom Function Fitting (Damped Oscillator)")
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
"""
## 4. 真实数据拟合案例

我们将读取 `longley_economic_data.csv`，并尝试拟合其中的两个变量。

"""

# %%
import pandas as pd

# 读取数据
data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/longley_economic_data.csv'
dataframe = pd.read_csv(data_path)
data = dataframe.values

# 提取第 6 列 (GNP.deflator) 作为 x，最后一列 (Employed) 作为 y
x_real, y_real = data[:, 5], data[:, -1]

# 尝试用 1阶、2阶、3阶多项式拟合
plt.figure(figsize=(10, 6))
plt.scatter(x_real, y_real, color='black', label='Real Data')

colors = ['red', 'green', 'blue']
for i in range(1, 4):
    # 拟合
    coeffs = np.polyfit(x_real, y_real, i)
    p = np.poly1d(coeffs)
    
    # 生成平滑的 x 用于绘图
    x_smooth = np.linspace(min(x_real), max(x_real), 100)
    plt.plot(x_smooth, p(x_smooth), color=colors[i-1], label=f'Degree {i} Fit')

plt.title("Fitting Real Economic Data")
plt.xlabel("GNP Deflator")
plt.ylabel("Employed")
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
"""
### 综合检测：放射性衰变曲线拟合

现在，轮到你亲自上手进行自定义函数拟合了！

**背景：**
在核物理实验中，某放射性同位素的剩余数量 $N(t)$ 随时间 $t$ 呈现指数衰减规律：
$$ N(t) = N_0 e^{-\lambda t} $$
其中 $N_0$ 是初始数量，$\lambda$ 是衰变常数。

**你的工程师任务：**
根据实验记录的散点数据 `t_data`（实验时间）和 `N_data`（衰变剩余量），请你完成以下步骤：
1. **定义模型函数** `decay_model`，注意 Python 中的指数函数应使用 NumPy 提供的 `np.exp()` 方法。
2. 使用 `scipy.optimize.curve_fit` 对定义好的模型以及样本数据进行拟合。提取出该函数第一个返回结果（即优化参数构成的元组/数组 `popt`）。

"""

# %%
import numpy as np
from scipy.optimize import curve_fit

# 实验测得某放射性物质的衰变数据
t_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
N_data = np.array([100.0, 60.5, 36.8, 22.3, 13.5, 8.2])

# TODO 1: 定义该放射性物质衰变的数学模型 N(t)
def decay_model(t, N0, lam):
    # 你的实现代码
    pass
    
# TODO 2: 使用优化器求得回归模型最优参数
popt, pcov = ...

N0_fit, lam_fit = popt
print(f"拟合出的初始剂量 N0: {N0_fit:.1f}")
print(f"拟合出的衰变常数 lambda: {lam_fit:.3f}")
