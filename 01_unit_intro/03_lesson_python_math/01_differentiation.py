# %% [markdown]
"""
# Python 数学计算与工程建模

大家好！在掌握了 Python 和 NumPy 的基础之后，我们现在要进入将编程应用于**数学建模与科学计算**的核心领域。

Python 之所以在科学计算和工程界（如电气工程、机械工程、航空航天）如此受欢迎，很大程度上归功于其强大的第三方库生态系统。其中最核心的库有：

1. **NumPy**: 我们已经见过，它提供了高性能的多维数组（`ndarray`）以及极速的矩阵运算。这是 Python 科学计算的基石。
2. **SciPy (Scientific Python)**: 这是建立在 NumPy 之上的“巨无霸”库。它提供了大量用于数学、科学和工程的算法，包括：优化、线性代数、积分、插值、特殊函数、快速傅里叶变换（FFT）、信号处理和图像处理等。
3. **SymPy**: 这是一个用于**符号数学计算**的库。它可以像你在纸上推导公式一样，进行精确的代数变形和微积分求解。

![Calculus in Physics](https://images.saymedia-content.com/.image/t_share/MTc0NDkwNjcyMTE0NDQzNjI0/how-to-understand-calculus-a-beginners-guide-to-differentiation-and-integration.jpg)

**本节课你将学到：**

1. **导数计算**: 包括符号求导和偏导数。
2. **工程应用**: 为什么计算机需要这两种不同的策略。

"""

# %% [markdown]
"""
## 导入工程计算的“兵器库”

在进行科学计算之前，我们首先需要导入相应的 Python 库。

"""

# %%
# 导入绘图库，用于将数据图形化
import matplotlib.pyplot as plt

# 导入符号计算库 SymPy
import sympy as sp

# 导入 NumPy，科学计算的核心库
import numpy as np

# 我们接下来课程还会用到的 SciPy 模块
from scipy.misc import derivative
from scipy import integrate
from scipy import optimize

print("科研计算环境初始化完成！")




# %% [markdown]
"""
## 1. 符号微分 (Symbolic Differentiation)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Sympy_logo.svg/300px-Sympy_logo.svg.png)

什么是**符号计算**？
在传统的编程中，如果你写 `y = x**2`，计算机必须先知道 `x` 的具体数值（比如 `x=3`），然后计算出 `y=9`。这个过程被称为**数值计算**。

但在**符号计算**中，`x` 不是一个具体的数字，而是一个**代数符号**（就像你在纸上写数学题一样）。`SymPy` 库运用微积分的基本法则（如链式法则、乘积法则等），对由符号表示的数学表达式进行完全精确的求导！

在复杂的系统建模初期，我们需要推导非线性动力学方程（如机器人的雅可比矩阵推导），SymPy 可以极大地避免人工推导的符号错误。

"""

# %%
# 定义一个符号变量 'x'
x = sp.Symbol('x')

# 定义一个关于 x 的函数 y = x^2 + sin(x)
y = x**2 + sp.sin(x)

# 使用 .diff() 方法对 y 关于 x 求一阶导数
y_prime = y.diff(x)

print("原函数 y =", y)
print("一阶导数 y' =", y_prime)

# 求二阶导数 (加速度)
y_double_prime = y.diff(x, 2)
print("二阶导数 y'' =", y_double_prime)




# %% [markdown]
"""
### 偏微分 (Partial Derivatives)

在多变量系统中（例如不仅与位置 `x` 有关，还与时间 `t` 或温度 `T` 有关的模型），我们需要对多变量函数求偏导。SymPy 处理这些同样轻而易举。

例如，一个简化的电路功率发热函数 $P(I, R) = I^2 R$。
我们可以求 $\frac{\partial P}{\partial I}$ 和 $\frac{\partial P}{\partial R}$。的表达式。

"""

# %%
# 同时定义两个符号变量 I (电流) 和 R (电阻)
I, R = sp.symbols('I R')

# 定义多元函数 P(I, R) = I^2 * R
P = (I**2) * R

# 对 I 求偏导数（表示电流微小变化对功率的影响）
dP_dI = P.diff(I)

# 对 R 求偏导数（表示电阻微小变化对功率的影响）
dP_dR = P.diff(R)

print("原函数 P(I, R) =", P)
print("对 I 的偏导数 dP/dI =", dP_dI)  # 理论上应该是 2*I*R
print("对 R 的偏导数 dP/dR =", dP_dR)  # 理论上应该是 I**2




# %% [markdown]
"""
### 实际小练：推导弹簧振子动力学方程

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Simple_harmonic_oscillator.gif/200px-Simple_harmonic_oscillator.gif)

假设一个轻弹簧振子系统的势能公式为 $U(x) = \frac{1}{2} k x^2$，动能公式为 $K(v) = \frac{1}{2} m v^2$。
已知保守力 $F$ 是势能 $U$ 对位置 $x$ 的负导数，即 $F = - \frac{dU}{dx}$。

请使用 SymPy 定义符号并计算出力的表达式 `F`，并在给定代码编辑器内完成计算。

"""

# %%
import sympy as sp

# 1. 定义符号变量 k 和 x
k, x = sp.symbols('k x')

# 2. 定义势能 U
U = 0.5 * k * x**2

# 3. 计算力 F (注意前面有负号)
F = # TODO: 填入计算代码

print("恢复力 F 的表达式为:", F)

# %% [markdown]
"""
### 随堂回顾

关于 **符号微分(如 SymPy 的实现)**，以下哪项描述是 **正确** 的？


**选项:**
- 它返回的是一个具体的数值，如 5.5，而不是一个函数表达式
- 它返回的是一个精确的导函数表达式，就像人工推断一样
- 它的计算速度比数值微分快得多，因此在实际仿真循环中总是首选
- 它只能用于线性函数，无法计算包含正弦、余弦或指数的函数

**正确答案:** 它返回的是一个精确的导函数表达式，就像人工推断一样
**提示:** 回顾一下我们刚才用SymPy求导的结果，输出的形式是公式（代数符号）。同时，符号计算由于需要底层代数引擎，速度其实远慢于数值计算的直接加减乘除。

"""
