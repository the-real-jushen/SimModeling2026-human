# %% [markdown]
"""
# 3. 子图 (Subplots) 与多图布局

有时候把所有曲线画在同一张图里会显得非常拥挤混乱。这时，我们可以使用 `plt.subplot()` 将画布划分为网格，并在不同的网格中绘制不同的图。

`plt.subplot(nrows, ncols, index)` 函数接受三个参数：
1.  `nrows`: 网格的行数。
2.  `ncols`: 网格的列数。
3.  `index`: 当前激活的子图索引（从 1 开始，从左到右，从上到下编号）。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0, 2*math.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 创建一个新的画布，并设置大小
plt.figure(figsize=(10, 8))

# --- 第一个子图：占据 2行1列 网格的第 1 个位置 (上半部分) ---
plt.subplot(2, 1, 1) 
plt.plot(x, y_sin, 'r-') # 'r-' 是 color='red', linestyle='-' 的简写
plt.title('Sine Wave')
plt.grid(True)

# --- 第二个子图：占据 2行1列 网格的第 2 个位置 (下半部分) ---
plt.subplot(2, 1, 2)
plt.plot(x, y_cos, 'b--') # 'b--' 是 color='blue', linestyle='--' 的简写
plt.title('Cosine Wave')
plt.grid(True)

# 自动调整子图之间的间距，防止标签重叠
plt.tight_layout() 
plt.show()




# %% [markdown]
"""
## 4. 散点图 (Scatter Plot)

散点图与折线图类似，但它不使用线将数据点连接起来。它非常适合用于展示离散数据的分布情况，或者寻找两个变量之间的相关性。

`plt.scatter(x, y, s=size, c=color)` 函数中：
*   `s` 可以是一个数值，也可以是一个与数据等长的数组，用于控制每个点的大小。
*   `c` 可以是一个颜色字符串，也可以是一个数值数组，配合 `cmap` (颜色映射) 来表示第三个维度的数值。

"""

# %%
# 生成两组服从二维正态分布的随机数据
# 第一组：均值(1,1)，方差较小
dots1 = np.random.multivariate_normal((1, 1), ((0.3, 0), (0, 0.3)), 50)
# 第二组：均值(3,3)，方差较大
dots2 = np.random.multivariate_normal((3, 3), ((1, 0), (0, 1)), 50)

plt.figure(figsize=(8, 6))

# 绘制第一组数据
# c='r' 设置颜色为红色
# s=dots1[:, 0]*30 让点的大小与它的 x 坐标值成正比
plt.scatter(dots1[:, 0], dots1[:, 1], c='r', s=dots1[:, 0]*30, alpha=0.6, label='Group A')

# 绘制第二组数据
# c='g' 设置颜色为绿色
# s=dots2[:, 1]*20 让点的大小与它的 y 坐标值成正比
plt.scatter(dots2[:, 0], dots2[:, 1], c='g', s=dots2[:, 1]*20, alpha=0.6, label='Group B')

plt.title("Scatter Plot with Variable Sizes")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.grid(True, linestyle='--')
plt.show()




# %% [markdown]
"""
## 5. 3D 曲面图 (3D Surface Plot)

为了绘制 3D 图形，我们需要使用面向对象（Object-Oriented）的绘图方式。我们首先创建一个 `Figure` 对象，然后向其中添加一个指定 `projection='3d'` 的 `Axes` 对象。

"""

# %%
# 导入 3D 绘图工具包 (虽然代码中没有直接调用它，但必须导入才能使用 '3d' 投影)
from mpl_toolkits.mplot3d import Axes3D

# 1. 准备网格数据
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
# np.meshgrid 将一维的坐标轴向量转化为二维的坐标矩阵
X, Y = np.meshgrid(x, y)

# 2. 计算 Z 轴数据 (一个二维高斯函数)
Z = X * np.exp(-X**2 - Y**2)

# 3. 创建画布和 3D 坐标系
fig = plt.figure(figsize=(10, 7))
# gca() 意思是 Get Current Axes，获取当前的坐标系实例
ax = fig.gca(projection='3d')

# 4. 绘制 3D 曲面
# cmap='viridis' 指定了颜色映射方案，使得不同高度显示不同颜色
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# 添加颜色条 (Colorbar) 以解释颜色与 Z 值的对应关系
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_title('3D Surface Plot of $z = x \cdot e^{-x^2 - y^2}$')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()




# %% [markdown]
"""
## 6. 等高线图 (Contour Plot)

我们使用与上面 3D 曲面图相同的 $X, Y, Z$ 数据，使用 `plt.contour()` 或 `plt.contourf()`（带颜色补充）来绘制。

"""

# %%
plt.figure(figsize=(8, 6))
# levels 参数可以指定绘制多少个等高线层级
contour = plt.contour(X, Y, Z, levels=20, cmap='RdBu')
plt.clabel(contour, inline=True, fontsize=8) # 在等高线上标注数值
plt.title('Contour Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()




# %% [markdown]
"""
## 7. 非线性坐标轴 (Logarithmic Scales)

我们可以将原本匀速递增的 `x` 或 `y` 坐标轴替换为指数/对数缩放体系。

"""

# %%
# 构造一些数据
x_log = np.arange(0, 100, 1)
y_log = x_log / 100

plt.figure(figsize=(10, 6))

# linear
plt.subplot(221)
plt.plot(x_log, y_log)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x_log, y_log)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x_log, y_log - y_log.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x_log, y_log)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
plt.show()




# %% [markdown]
"""
## 8. 极坐标系 (Polar Coordinates)

向 `subplot` 索取画板时传入 `projection='polar'` 即可切入极坐标模式。这时的输入不再是直角坐标而是 角度 $\theta$ 与 半径 $r$。

"""

# %%
# 随机生成用于极坐标系的数据点
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

plt.figure(figsize=(6, 6))
# 抓取一个极坐标系的背景对象
ax = plt.subplot(111, projection='polar')

# 画上极坐标散点
ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
ax.set_title('Polar Scatter Plot', va='bottom')
plt.show()




# %% [markdown]
"""
### 知识连线题：子图的排布网格

让我们回测一下关于子图网格布局的理解。如果我想创建一个 **2 行 x 3 列** 的网格子图结构，且我当前的绘图焦点想要放置在整个坐标系的**左下角第一个图（也就是第 2 行的第 1 列）**，我应该使用以下哪句代码？


**选项:**
- `plt.subplot(1, 2, 3)`
- `plt.subplot(3, 2, 6)`
- `plt.subplot(2, 3, 4)`
- `plt.subplot(2, 3, 2)`

**正确答案:** `plt.subplot(2, 3, 4)`
**提示:** 前两个数字分别代表行数(2)和列数(3)。由于编号是从左到右，从上到下，因为每一行有3个位置，所以第 2 行的首列恰好就是顺数的第 4 个位置（3+1=4）。

"""

# %% [markdown]
"""
### 综合挑战：弹跳小球的物理轨迹可视化编程

一个小球，在平地上从高度为 `h` 的某处**水平**抛出，其初始水平速度为 `v`。
当其落地并弹起瞬间的物理特性为：
1. **水平方向无摩擦**：水平速度保持绝对不变连续匀速运作。
2. **垂直碰撞动能衰损**：碰撞导致的动能折损会让它弹起时的“竖直方向上升初速度”变为刚刚砸地瞬间“落地垂直下落尾速度”的 **0.7** 倍。
   
请结合物理学的基础下落公式，在对应的 Python 函数内利用 `numpy` 构建时间和位移向量，调用 `matplotlib.pyplot` 帮助我们画出这个小球从刚刚脱手平抛，一直到**第二次**砸落地盘为止的完整 $x-y$ 运动曲线图（分为两段抛物线）。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def bounce_ball(h, v):
    g = 9.8
    # TODO: 请在这个函数里利用 numpy 向量和 matplotlib 功能
    # 画出平抛小球完整经历了第一次反弹(弹起垂直回升初速度衰减为 0.7 倍，水平速度 v 不变)
    # 直至第二次砸向地面的完整两段连续抛物线轨迹。
    pass
    
bounce_ball(10, 10)
