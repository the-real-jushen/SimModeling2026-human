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

#%% [markdown]
"""

## 3. 综合挑战练习

下面这道题比前面的例子更接近真实工程场景。
你需要自己生成数据，并把多种图表元素组合起来完成一个较完整的可视化任务。


### 挑战题：分析带噪声的传感器信号

某传感器理论上测得的是一个平滑的周期信号，但实际采集时会混入随机噪声。
请你自己生成一组“理论信号”和“带噪声的测量信号”，并用 Matplotlib 做出清晰的可视化分析图。

**已知条件：**
1. 时间范围取 $0$ 到 $4\pi$。
2. 用 `np.linspace()` 生成 200 个时间点。
3. 理论信号设为：
    $$ y_{ideal} = \sin(x) + 0.3\sin(3x) $$
4. 噪声可以用 `np.random.normal()` 生成。
5. 测量信号设为：
    $$ y_{measured} = y_{ideal} + noise $$

**任务要求：**
1. 在同一个代码块中生成 `x`、`y_ideal`、`noise`、`y_measured`。
2. 创建一个包含 2 个子图的画布，建议使用 `plt.subplot(2, 1, ...)`。
3. 第 1 个子图中：
    - 画出理论信号 `y_ideal`
    - 再画出测量信号 `y_measured`
    - 两条曲线要有不同颜色、线型或透明度
    - 添加标题、坐标轴标签、图例和网格线
4. 第 2 个子图中：
    - 画出误差曲线 `error = y_measured - y_ideal`
    - 给误差曲线设置不同的颜色
    - 添加标题、坐标轴标签和网格线
5. 调整整张图的布局，避免标题或标签重叠。

**扩展挑战（选做）：**
1. 尝试用散点图显示测量信号，而不是普通折线图。
2. 给误差图加一条 `y=0` 的参考线。
3. 修改噪声大小，观察图像变化。

**提示：**
1. 如果你希望每次运行结果一致，可以先写 `np.random.seed(0)`。
2. 布局调整可以尝试 `plt.tight_layout()`。
3. 如果测量信号看起来太乱，可以适当减小噪声的标准差。

"""

# %%
# TODO: 请在下面完成这道综合挑战题
# 建议步骤：
# 1. 生成 x
# 2. 生成理论信号 y_ideal
# 3. 生成随机噪声 noise
# 4. 得到测量信号 y_measured
# 5. 计算误差 error
# 6. 用两个子图完成可视化


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
# add_subplot() 会创建一个新的 3D 坐标系，兼容性比 gca(projection='3d') 更好
ax = fig.add_subplot(111, projection='3d')

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
## 9. `projection` 还有哪些常见选项？

在 Matplotlib 中，`projection` 用来告诉坐标轴：要用什么“坐标系/投影方式”来显示数据。

下面是几种常见类型：

1. `rectilinear`
    - 默认的直角坐标系，也就是最常见的 x-y 平面。
    - 如果你不写 `projection`，通常就是它。

2. `polar`
    - 极坐标系。
    - 适合表示角度与半径，例如雷达方向、旋转系统、周期分布等。

3. `3d`
    - 三维坐标系。
    - 适合曲面图、三维散点图、空间轨迹图。

4. `aitoff`
    - 一种地图投影方式。
    - 常用于展示全球经纬度数据。

5. `hammer`
    - 也是地图投影方式。
    - 常用于全球范围分布图，可视化效果较平滑。

6. `lambert`
    - Lambert 地图投影。
    - 也可用于球面方向数据的展示。

7. `mollweide`
    - Mollweide 地图投影。
    - 常用于全球数据、天文数据的整体分布图。

注意：
- `polar`、`aitoff`、`hammer`、`lambert`、`mollweide` 都可以直接通过 `subplot(..., projection='...')` 创建。
- `3d` 更推荐用 `fig.add_subplot(..., projection='3d')`。

"""

# %%
# 演示几种常见 projection 的效果

fig = plt.figure(figsize=(14, 10))

# 1. 默认直角坐标系 rectilinear
ax1 = fig.add_subplot(2, 3, 1)
x_demo = np.linspace(0, 2*np.pi, 200)
ax1.plot(x_demo, np.sin(x_demo), color='tab:blue')
ax1.set_title('rectilinear')
ax1.grid(True)

# 2. 极坐标 polar
ax2 = fig.add_subplot(2, 3, 2, projection='polar')
theta_demo = np.linspace(0, 2*np.pi, 400)
r_demo = 1 + 0.4 * np.sin(5 * theta_demo)
ax2.plot(theta_demo, r_demo, color='tab:orange')
ax2.set_title('polar')

# 3. Aitoff 投影
ax3 = fig.add_subplot(2, 3, 3, projection='aitoff')
lon = np.linspace(-np.pi, np.pi, 400)
lat = 0.4 * np.sin(2 * lon)
ax3.plot(lon, lat, color='tab:green')
ax3.set_title('aitoff')
ax3.grid(True)

# 4. Hammer 投影
ax4 = fig.add_subplot(2, 3, 4, projection='hammer')
ax4.plot(lon, lat, color='tab:red')
ax4.set_title('hammer')
ax4.grid(True)

# 5. Lambert 投影
ax5 = fig.add_subplot(2, 3, 5, projection='lambert')
ax5.plot(lon, lat, color='tab:purple')
ax5.set_title('lambert')
ax5.grid(True)

# 6. Mollweide 投影
ax6 = fig.add_subplot(2, 3, 6, projection='mollweide')
ax6.plot(lon, lat, color='tab:brown')
ax6.set_title('mollweide')
ax6.grid(True)

plt.tight_layout()
plt.show()


# %%
# 单独演示 3d projection

# 构造三维曲面数据
x3d = np.linspace(-2, 2, 80)
y3d = np.linspace(-2, 2, 80)
X3d, Y3d = np.meshgrid(x3d, y3d)
Z3d = np.sin(np.sqrt(X3d**2 + Y3d**2))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X3d, Y3d, Z3d, cmap='viridis', edgecolor='none', alpha=0.85)

ax.set_title('3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()

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
    # 第一段：小球从高度 h 水平抛出，初始竖直速度为 0
    t1_end = np.sqrt(2 * h / g)
    t1 = np.linspace(0, t1_end, 200)
    x1 = v * t1
    y1 = h - 0.5 * g * t1**2

    # 第一次落地前的竖直速度大小，用于计算反弹后的初始上抛速度
    vy_before_bounce = g * t1_end
    vy_after_bounce = 0.7 * vy_before_bounce

    # 第二段：从第一次落地点开始，继续运动直到第二次落地
    # 从地面竖直向上抛出，起始高度为 0
    t2_end = 2 * vy_after_bounce / g
    t2 = np.linspace(0, t2_end, 200)
    x2 = x1[-1] + v * t2
    y2 = vy_after_bounce * t2 - 0.5 * g * t2**2

    plt.figure(figsize=(10, 6))

    # 两段轨迹分别画出，便于观察反弹前后变化
    plt.plot(x1, y1, label='First flight')
    plt.plot(x2, y2, label='After first bounce')

    # 标出起点、第一次落地点和第二次落地点
    plt.scatter([0, x1[-1], x2[-1]], [h, 0, 0], color='red', zorder=5)

    plt.title('Bouncing Ball Trajectory')
    plt.xlabel('Horizontal Distance x (m)')
    plt.ylabel('Height y (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.show()
    
bounce_ball(10, 10)

# %%
