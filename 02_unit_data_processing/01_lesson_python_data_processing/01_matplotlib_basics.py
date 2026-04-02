# %% [markdown]
"""
# Python 进阶与数据处理：数据可视化基础

大家好！在数学建模和工程应用中，我们经常需要处理大量的数据。仅仅看着一堆数字是很难发现规律的，因此**数据可视化**（Data Visualization）成为了不可或缺的技能。

在 Python 生态中，最基础、最强大的绘图库就是 **Matplotlib**。它的设计灵感来源于 MATLAB 的绘图接口，因此如果你有 MATLAB 经验，会觉得非常熟悉。

![Matplotlib Plotting](https://upload.wikimedia.org/wikipedia/commons/1/17/Matplotlib_screenshot.png)
*(图示：Matplotlib 可以绘制各种复杂、精美的科学图表)*

**本节课你将学到：**

1.  如何使用 `matplotlib.pyplot` 绘制基本的二维曲线图。
2.  如何定制图表的各种元素（颜色、线型、标记、标签、图例等）。
3.  如何在一张画布上绘制多个子图（Subplots）。
4.  如何绘制散点图（Scatter plots）和简单的 3D 曲面图。

"""

# %% [markdown]
"""
## 1. 基本的绘图操作

我们从最简单的二维曲线图开始。要画一条曲线，我们只需要提供一系列的 x 坐标和对应的 y 坐标。

"""

# %%
# 导入 NumPy 用于生成数据
import numpy as np
# 导入 Matplotlib 的 pyplot 模块用于绘图
import matplotlib.pyplot as plt
import math

# 生成 x 轴数据：从 0 到 2*pi，均匀生成 50 个点
x = np.linspace(0, 2*math.pi, 50)

# 计算对应的 y 轴数据：y = sin(x)
y = np.sin(x)

# 使用 plt.plot() 绘制曲线
plt.plot(x, y)

# 添加标题
plt.title("Basic Sine Wave")

# 显示图表 (在 Jupyter/交互式环境中通常会自动显示，但在纯脚本中必须调用 show)
plt.show()




# %% [markdown]
"""
### 在同一张图上绘制多条曲线

我们可以很容易地将多条曲线叠加在一起进行对比。

"""

# %%
# 计算第二条曲线的数据：y2 = sin(x + pi/2) = cos(x)
y2 = np.sin(x + math.pi/2)

# 绘制第一条曲线
plt.plot(x, y, label='sin(x)')

# 绘制第二条曲线
plt.plot(x, y2, label='cos(x)')

# 显示图例 (Legend)，它会根据 plot 函数中的 label 参数自动生成
plt.legend()

plt.title("Multiple Curves on One Plot")
plt.show()




# %% [markdown]
"""
## 2. 定制图表元素

为了让图表更具可读性和专业性，我们需要掌握如何调整图表的各种属性。

以下是一些常用的 `plt.plot()` 参数和图表设置函数：

*   **`color` (或 `c`)**: 线的颜色（如 'red', 'b', '#FF0000'）。
*   **`linestyle` (或 `ls`)**: 线的样式（如 '-', '--', ':', '-.'）。
*   **`linewidth` (或 `lw`)**: 线的宽度。
*   **`marker`**: 数据点的标记形状（如 'o' 圆圈, '*' 星号, '^' 三角形）。
*   **`alpha`**: 透明度，范围 0 (完全透明) 到 1 (完全不透明)。
*   **`plt.xlim()`, `plt.ylim()`**: 设置 x 轴和 y 轴的显示范围。
*   **`plt.xlabel()`, `plt.ylabel()`**: 设置坐标轴的标签。
*   **`plt.grid()`**: 显示背景网格线。

"""

# %%
# 重新生成数据，这次点少一点以便看清 marker
x_sparse = np.linspace(0, 2*math.pi, 15)
y_sparse = np.sin(x_sparse)

plt.figure(figsize=(8, 5)) # 设置画布大小

# 综合使用各种参数定制曲线
plt.plot(x_sparse, y_sparse, 
         alpha=0.7,                 # 透明度 70%
         label='Sampled Sine',      # 图例标签
         color='purple',            # 线条颜色为紫色
         linestyle='-.',            # 线型为点划线
         linewidth=2,               # 线宽为 2
         marker='D',                # 标记形状为菱形 (Diamond)
         markersize=8,              # 标记大小
         markerfacecolor='yellow',  # 标记内部填充颜色
         markeredgecolor='black')   # 标记边缘颜色

# 设置坐标轴范围
plt.xlim([0, 2*math.pi])
plt.ylim([-1.2, 1.2])

# 设置标题和坐标轴标签
plt.title('Highly Customized Plot', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (V)', fontsize=12)

# 开启网格线，并设置为虚线
plt.grid(True, linestyle=':')

# 显示图例，放置在最佳位置
plt.legend(loc='best')

plt.show()




# %% [markdown]
"""
### 彩蛋：你可以画很多有趣的东西 (Just for fun)

让我们利用数学公式画一个“心”形图案。
参数方程为：
$$x = 16 \sin^3(t)$$
$$y = 13 \cos(t) - 5 \cos(2t) - 2 \cos(3t) - \cos(4t)$$

"""

# %%
# just for fun!
t = np.linspace(0, 2*np.pi, 100)

x3 = 16 * np.sin(t)**3
y3 = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

plt.figure(figsize=(6, 6))
# 使用 RGB tuple 配置颜色，设置极宽的线条
plt.plot(x3, y3, c=(1, 0.2, 0.5), lw=20)

plt.title('Heart!')
plt.axis('equal') # 保持 x 和 y 轴比例一致，以免心型被拉伸变形
plt.axis('off')   # 关闭坐标轴显示
plt.show()




# %% [markdown]
"""
### 知识检查

在 Matplotlib 中，哪个函数用于在图表中显示解释不同曲线含义的图例框？


**选项:**
- plt.title('My Plot')
- plt.xlabel('X-axis')
- plt.legend()
- plt.grid(True)

**正确答案:** plt.legend()
**提示:** 图例（Legend）是用来解释图中不同颜色或线型代表什么数据的框。它依赖于 `plot()` 函数中的 `label` 参数。

"""

# %% [markdown]




