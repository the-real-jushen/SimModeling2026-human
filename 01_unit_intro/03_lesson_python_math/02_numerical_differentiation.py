# %% [markdown]
"""
# 2. 数值微分 (Numerical Differentiation)

在上一节我们见识了牛逼的 SymPy **符号求导**。但在实际的电气工程和物理仿真中，我们绝大多数时候**不用它**！

为什么？
因为真实世界的数据往往没有完美的数学公式！
比如：
- 气象站传感器的风速读数 `[2.1, 2.3, 2.5, 3.1]`。你连函数公式都没有，怎么求导（求加速度）？
- 你正在训练一个有 100 亿个参数的神经网络，如果调用 SymPy 去推导一个两页纸长的公式再代入数据，计算机一万年也算不完。

这时候我们就必须依赖**数值微分**：只基于具体的数据点，用离散的加减乘除，近似逼近真实的导数值。

## 差分 (Difference)

数值微分的核心思想来源于高中的导数定义：
$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$

因为计算机没法让 $h$ 真正趋于 0，所以我们就硬塞哪怕很小的 $h$ (比如 `0.01`或测量时间间隔 $\Delta t$) 给它算：
导数 $\approx \frac{\Delta y}{\Delta x}$

在 NumPy 中，这就叫**差分** (Difference)。NumPy 提供了极快的方法来计算相邻数据点之间的差值：`np.diff()`。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import math

# 假设这是一辆汽车每秒记录的位置 (米)
position = [0, 2, 8, 18, 32, 50]
dt = 1.0 # 采样时间间隔 1秒

# np.diff() 会计算相邻两个元素的差值 [2-0, 8-2, 18-8, 32-18, 50-32]
delta_position = np.diff(position)

# 速度 = 距离差 / 时间差
velocity = delta_position / dt

print("位置序列:", position)
print("相邻位置差(位移区间):", delta_position)
print("速度近似值:", velocity)

# 请注意！！！
# 原序列有 6 个点，差分之后只剩下 5 个点了！因为 6 个点之间只有 5 个线段。




# %% [markdown]
"""
## 梯度 (Gradient)：多维空间的导数

在解决优化问题（比如寻找最小功耗的电路参数，或者训练人工智能）时，我们经常要面对地形像“山峰和山谷”的多变量函数。

**梯度 (Gradient)** 就是多元函数的“导数排成的向量”。
梯度的几何意义非常直白：**它永远指向这座山上，当前位置最陡峭的上坡方向！**

![Gradient Vector Field](https://calcworkshop.com/wp-content/uploads/gradient-vector-field-level-curves.png)
*(红色箭头表示梯度向量，它们总是垂直于等高线，指向函数值增大的方向。)*

NumPy 的 `np.gradient()` 函数比 `np.diff()` 更聪明。为了保证导数的“对齐”，它默认使用**中心差分法**计算，所以求出来的导数数组，**长度和原来的数组一模一样！**

"""

# %%
# 我们用代码重现上面的等高线与梯度图
# 1. 铺设一张 2D 网格(像地图一样)
x = np.arange(-2, 2, 0.2)
y = np.arange(-2, 2, 0.2)
X, Y = np.meshgrid(x, y)

# 2. 创造一座“地形山”，高度 Z 和 X, Y 有关: Z = X * e^(-X^2 - Y^2)
Z = X * np.exp(-X**2 - Y**2)

# 3. 计算二维地形的梯度
# np.gradient 会返回 Y 陡峭度(南北向斜率) 和 X 陡峭度(东西向斜率)
grad_y, grad_x = np.gradient(Z, 0.2, 0.2)  # 0.2 是网格间距

# ==== 绘图代码 ====
plt.figure(figsize=(8, 6))
# 画高度的“等高线图”
contour = plt.contour(X, Y, Z, levels=15, cmap='viridis')
# 画梯度箭头 (Quiver)
plt.quiver(X, Y, grad_x, grad_y, color='red', alpha=0.6)

plt.title("二维地形的梯度场 (红色箭头永远指向最高处)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()




# %% [markdown]
"""
### 随堂测试

关于数值微分和梯度计算，下列说法 **正确** 的是？

"""
