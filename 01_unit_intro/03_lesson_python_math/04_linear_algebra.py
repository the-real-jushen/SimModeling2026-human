# %% [markdown]
"""
# 4. 线性代数与线性方程组求解

线性代数是现代工程、科学计算以及人工智能的绝对基石。
在电气工程中，无论是基尔霍夫电路定律列出的节点电压方程组，还是电网潮流计算中的雅可比矩阵，最终**全部都会转化为矩阵和线性方程组求解**。

NumPy 提供了一个极为强大的线性代数子模块：`numpy.linalg` (Linear Algebra的缩写)。

![Matrix Form](https://educatemath.com/wp-content/uploads/2022/07/general-form-of-system-of-linear-equations-matrix-form-1-800x450.jpg)
*(任何由加法和常数乘法组成的多元方程组，都可以抽象为清爽的矩阵相乘方程：$Ax = b$)*

"""

# %% [markdown]
"""
## 求解适定方程组 (There is exactly one solution)

假设我们在分析一个包含两个未知节点电压 $v_1, v_2$ 的电路，根据基尔霍夫定律得到了如下方程组：
$$ 1 \cdot v_1 + 2 \cdot v_2 = 1 $$
$$ 3 \cdot v_1 + 5 \cdot v_2 = 2 $$

将其转化为矩阵形式 $Ax = b$：
$$
\begin{bmatrix} 1 & 2 \\ 3 & 5 \end{bmatrix}
\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} =
\begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

只要系数矩阵 $A$ 是**满秩的方阵**（即方程个数等于未知数个数，且互相独立不矛盾），我们就可以使用 `np.linalg.solve(A, b)` 瞬间求出精确解。

"""

# %%
import numpy as np

# 定义系数矩阵 A (2x2)
A = np.array([
    [1, 2],
    [3, 5]
])

# 定义常数向量 b
b = np.array([1, 2])

# 使用 linalg.solve 直接求解
# 警告：不要自己去用 np.linalg.inv(A) 求逆矩阵然后再乘以 b。
# 数值计算中，求逆极其耗时且容易产生浮点数精度爆炸。solve() 底层使用了更稳定的LU分解！
x = np.linalg.solve(A, b)

print("节点电压解 [v1, v2]:", x)

# 我们可以验证一下 A @ x 是否等于 b (@ 符号是 Python 中矩阵乘法的简写)
print("验算结果 A @ x =", A @ x)




# %% [markdown]
"""
## 求解超定方程组：最优化与最小二乘法

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/400px-Linear_regression.svg.png)

在实验科学中，我们经常遇到**方程个数远大于未知数个数**的情况。
比如你在实验室测量一个线性电阻的电阻值 $R$ （根据欧姆定律 $U = R \cdot I$）。你测了 3 次，得到了三组 $(U, I)$ 数据，于是列出了 3 个方程。但是你只有一个未知数 $R$！
由于测量仪器的噪声误差，这 3 个方程是不可能同时完全成立的。

这就是**超定方程组 (Overdetermined System)**。
针对这种情况，我们不再追求“绝对相等的解”，而是寻找一个能让所有方程的“误差平方和最小”的妥协解——这就是大名鼎鼎的**最小二乘法 (Least Squares)**。

在 NumPy 中，使用 `np.linalg.lstsq()`。

"""

# %%
# 假设我们要拟合一条直线 y = m*x + c，我们需要求出 m 和 c 两个未知数。
# 但我们采集了 4 个数据点 (x, y)：(0, 1), (1, 3), (2, 4), (3, 7)
# 对应方程为：
# 0*m + 1*c = 1
# 1*m + 1*c = 3
# 2*m + 1*c = 4
# 3*m + 1*c = 7

# 我们将其写成矩阵形式 Ax = y
A = np.array([
    [0, 1],
    [1, 1],
    [2, 1],
    [3, 1]
])
y = np.array([1, 3, 4, 7])

# 使用 lstsq 求解最小二乘法解
# rcond=None 是为了保证跨版本兼容性的默认参数
result = np.linalg.lstsq(A, y, rcond=None)

# result 是一个包含 4 个元素的元组。第一个元素就是我们要的最优解 [m, c]。
x_best = result[0]
m, c = x_best

print(f"最优拟合直线斜率 m = {m:.2f}")
print(f"最优拟合直线截距 c = {c:.2f}")

# 第二个元素 result[1] 能够告诉我们拟合残差的总和 (误差有多大)
print(f"总残差平方和: {result[1]}")




# %% [markdown]
"""
### 随堂实战：构建电路矩阵方程求解

你正在设计一个三相电力系统。根据节点分析法，你得到了关于三个节点电压（V1, V2, V3）的方程组：

1. `4*V1 - 1*V2 + 0*V3 = 20`
2. `-1*V1 + 4*V2 - 1*V3 = -10`
3. `0*V1 - 1*V2 + 4*V3 = 30`

这是一个非常典型的具有“对称性”的问题（类似三对角矩阵）。
请你在下方的代码块中，构建矩阵 $A$ 和向量 $b$，并使用正确的 NumPy 函数求解节点电压。

"""

# %%
import numpy as np

# 1. 构建 3x3 的系数矩阵 A 
A = np.array([
    [4, -1, 0],
    # 补全后面的行...
    # TODO
])

# 2. 构建 1D 的结果向量 b
b = # TODO

# 3. 计算结果 V
V = # TODO

print("解得的电压 [V1, V2, V3] 为:", V)

# %% [markdown]
"""
### 随桌思考

如果你面对这样一个方程组：
1. `x + y = 3`
2. `x - 2y = 1`
3. `2x + y = 4`

以下关于该方程组在使用计算机求解时的说法，**正确**的是：


**选项:**
- `np.linalg.solve(A, b)` 是求解此类方程组的最佳工具。
- 我们应该先求 A 矩阵的逆矩阵 `inv(A)` 才能得出答案。
- 由于未知数只有 2 个，但方程有 3 个，所以不存在绝对完美的精确解，应使用 `np.linalg.lstsq` (最小二乘法)。
- 矩阵 A 必定是一个满秩的方阵。

**正确答案:** 由于未知数只有 2 个，但方程有 3 个，所以不存在绝对完美的精确解，应使用 `np.linalg.lstsq` (最小二乘法)。
**提示:** 仔细看，方程组包含 3 个方程，但只有 x, y 2个未知数。这叫超定方程组。

"""
