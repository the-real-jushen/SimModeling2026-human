# %% [markdown]
"""
# NumPy 简介：工程计算的坚实底座 🧱

在上一节，我们学习了 Python 的内置数据结构，比如列表（List）。
但在真正的工程仿真和数据科学中，Python 原生的列表存在一个**致命的弱点：太慢了**！

想象一下，你正在处理一个风洞测试采集到的数据，或者模拟一千万个粒子的运动轨迹。如果你使用 Python 原生的库和 `for` 循环来逐一计算每个粒子的新坐标，你的程序可能会运行几个小时。

**NumPy (Numerical Python)** 就是为了解决这根“软肋”而诞生的。它是 Python 科学计算生态体系中**最核心、最底层、最重要的基础库**。后续我们会学习的 Pandas (数据处理) 和 Matplotlib (绘图)，甚至深度学习框架 PyTorch，都在很大程度上借鉴或直接依赖了 NumPy 的设计思想。

"""

# %% [markdown]
"""
## 为什么原生列表不够用？为什么我们需要 NumPy？

**1. 内存碎片的梦魇**
Python 的列表是一个“大杂烩”，你可以在里面同时塞进整数、字符串、甚至另一个列表 `[1, "hello", 3.14, [1, 2]]`。为了实现这种非凡的灵活性，Python 在底层其实存放的是对象的**指针（引用）**，而不是真正的数据本身。当我们要遍历百万级数据时，计算机不得不到处在内存里“跳跃”寻找真实数据，导致极高的开销。

**2. 没有内置的数学运算支持**
如果你想让一个列表中所有的数字都乘以 2（模拟电压放大），原生的 Python 列表会直接变成 `[1, 2, 3] * 2 = [1, 2, 3, 1, 2, 3]` （列表复制粘贴拼接）。你必须写一个缓慢的 `for` 循环：`[x * 2 for x in my_list]`。

**NumPy 的解法：`ndarray` (多维数组)**
- **高度同质化**：一个数组里的所有元素必须是**同一种类型**（通常是 C 语言层级的 `float64` 或 `int32`）。
- **底层连续内存**：数据在内存里紧紧挨着，计算机可以直接进行批量的块读取（由底层的 C 语言高度优化）。
- **向量化运算 (Vectorization)**：你可以直接把整个数组当成一个单独的数字来写数学公式结构！底层的 C 语言会瞬间帮你完成循环并发计算。

![NumPy Array vs Python List](https://www.python51.com/wp-content/uploads/2021/04/1611904527811809.png)
*(原生 Python 列表里的每个元素自身都有厚重的对象外壳，而 NumPy 数组仅仅是纯粹的 C 数据块。)*

"""

# %%
import numpy as np
import time

# 让我们用一个直观的例子来证明 NumPy 的压倒性速度优势
# 假设我们要对一千万个气温读数进行摄氏度转华氏度： F = C * 1.8 + 32
size = 10000000

# 纯 Python 列表版本
python_list = list(range(size))
start_time = time.time()
python_result = [c * 1.8 + 32 for c in python_list]
end_time = time.time()
print(f"纯 Python 列表耗时: {end_time - start_time:.4f} 秒")

# NumPy 数组版本
numpy_array = np.arange(size)
start_time = time.time()
# 向量化运算！整齐利落，一句代码搞定
numpy_result = numpy_array * 1.8 + 32
end_time = time.time()
print(f"NumPy 向量化耗时: {end_time - start_time:.4f} 秒")
# 你会发现 NumPy 快了通常 10 倍以上！




# %% [markdown]
"""
## 创建 NumPy 数组的常见姿势 🛠️

在建模时，我们经常需要生成特定规律的数组，比如全 0 矩阵用来初始化状态，或者等差数列用来表示时间步长。NumPy 提供了非常方便的内置函数。

其中经常使用的便是类似原生 `range()` 的 `np.arange()`。这里需要特别强调 Python 中极其重要的 **区间生成与切片规则**：
无论是原生的 `range()`、NumPy 的 `np.arange()`，还是数组切片 `[start:stop:step]`，它们都高度统一地遵循 **“包含起点 (Start)，不包含终点 (Stop)”** 的半开半闭区间逻辑，并可以通过 **步长 (Step)** 控制增量。

你可以通过下面这张图直观地理解 `Start`, `Stop`, `Step` 的运作机制：

![](images/range.png)

"""

# %%
# 1. 从一维和二维 Python 列表装载
arr_1d = np.array([1, 2, 3, 4, 5])
print("一维数组 (Vector):\n", arr_1d)

# 💡注意规律：如果你尝试传入不同类型的数据，NumPy 会自动统一它们
arr_mixed = np.array([1, 2.5, "hello"])
print("\n类型自动转换 (变成字符串):\n", arr_mixed)

my_matrix = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(my_matrix)
print("\n二维数组 (Matrix):\n", arr_2d)
print(f"该二维数组的维度大小 (Shape) 是: {arr_2d.shape}")  # (2, 3) 意味着 2行 3列

# Reshape 重塑变换: 改变数组的形状而不改变其中的数据
# 比如把一个一维 6 元素数组重新变成 2 行 3 列 (-1 会自动推导需要的长度)
arr_flat = np.arange(0, 6)
arr_reshaped = arr_flat.reshape(2, -1)
print("\n原本的平铺数组:\n", arr_flat)
print("Reshape 自动计算列数后的数组:\n", arr_reshaped)

# 2. 快速生成特定矩阵的函数
zeros = np.zeros((3, 3))
print("\n3x3 全 0 矩阵 (常用于初始化状态变量):\n", zeros)

ones = np.ones((2, 4))
print("\n2x4 全 1 矩阵:\n", ones)

# 3. 时间步长的好帮手：arange 和 linspace
# np.arange(start, stop, step) -> 不包含 stop
seq = np.arange(0, 10, 2)
print("\n从 0 到 10 (步长2，不含10) 的数列:\n", seq)

# np.linspace(start, stop, num_of_elements) -> 包含 stop，极度常用！
# 比如我们要把 0秒 到 1秒 平均切分成 5 个时间点
time_steps = np.linspace(0, 1, 5)
print("\n0 到 1 之间均匀分布的 5 个时间点:\n", time_steps)




# %% [markdown]
"""
## 数组的数学魔法：广播 (Broadcasting) 📻

还记得我们前面提到的“向量化运算”吗？NumPy 允许你把整个数组像标量一样加减乘除。
更神奇的是，当你对**形状不同**的数组进行运算时，NumPy 会自动运用一套叫做**广播 (Broadcasting)** 的规则。

举个例子：你想把一个 $3 \times 3$ 的矩阵，加上一个包含三个元素的 $1 \times 3$ 行向量。在严格的线性代数里这是不被允许的，但 NumPy 知道你想做的是把这个行向量“复制粘贴”应用到矩阵的**每一行**上。

"""

# %%
# 创建一个 3x3 矩阵，代表 3 个传感器在 3 次采样中的读取电压
voltages = np.array([
    [1.1, 1.2, 1.0],
    [2.1, 2.2, 2.0],
    [3.1, 3.2, 3.0]
])

# 假设由于设备老化的标定误差，我们需要给每个传感器(每列)补上相应的偏置(Offsets)
offsets = np.array([0.5, 0.0, -0.5])

# 【广播魔法】直接相加！NumPy 自动把 offsets 扩展到每一行
calibrated_voltages = voltages + offsets

print("原始电压:\n", voltages)
print("\n偏置补偿:\n", offsets)
print("\n校准后的电压 (广播相加):\n", calibrated_voltages)

# 取出某一列、某一行的方法 (切片索引 Array Slicing)
# voltages[行, 列]，用 ':' 代表取所有内容
print("\n只看第 2 列的所有数据:")
print(calibrated_voltages[:, 1])




# %% [markdown]
"""
### 代码实战阶段：NumPy 数学物理公式向量化

现在，请你运用刚学到的 NumPy 工具完成一个物理计算小任务。

**任务场景：** 
你是一名土木工程师，你需要计算一座桥梁上 5 个不同位置的变形量 $D$。
根据简化模型，变形公式为：
$D = k \cdot \sin(x)$

已知刚度系数 $k = 10$，位置坐标向量 $x$ 为 `[0, 0.5, 1.0, 1.5, 2.0]`。
请使用 NumPy 提供的向量化运算和 `np.sin()` 函数，**不使用任何 for 循环**，一步求出五个位置的变形量数组 `D_array`。

*(提示：你只需要在占位符处填入一个正确的 NumPy 表达式即可)*

"""

# %%
import numpy as np

k = 10
# 定义坐标向量
x_array = np.array([0, 0.5, 1.0, 1.5, 2.0])

# 请在这里完成代码，填入单行公式计算 D_array：
D_array = ______

print("位置:", x_array)
print("变形量:", np.round(D_array, 3))

# %% [markdown]
"""
### 实战挑战：创建指定形状的 Numpy 数组

写代码生成下面这些数据，同样颜色的为一个nparray，注意维度和方向：

![](images/2021-03-26-23-59-29.png)

*(请运用所学的 NumPy 方法尝试还原图像中的矩阵/数组)*

"""

# %%
import numpy as np

# 请根据图片要求，分别创建这几个不同颜色代表的numpy数组
# 你可以自由使用 np.array, np.zeros, np.arange, reshape等函数

array_1 = # TODO
array_2 = # TODO

print(array_1)
print(array_2)

# %% [markdown]
"""
### 随堂回顾

基于我们今天对 NumPy 的了解，请选出以下表述 **正确** 的一项：


**选项:**
- `np.array([-1, 0, 1]) + 5` 会引发报错，因为加法只能发生在两个相同形状的数组之间
- `np.linspace(0, 10, 11)` 会生成从 0 到 10 包含首尾的 11 个均匀分布的数字
- Python 的原生的 `for` 循环由于经过了高度优化，遍历计算速度比 NumPy 向量化运算更快
- NumPy 数组可以像原生列表一样，将不同类型的数据混合存放在同一个维度中

**正确答案:** `np.linspace(0, 10, 11)` 会生成从 0 到 10 包含首尾的 11 个均匀分布的数字
**提示:** 回顾一下我们关于内存机制、向量化以及 `linspace` 函数的讲解。

"""
