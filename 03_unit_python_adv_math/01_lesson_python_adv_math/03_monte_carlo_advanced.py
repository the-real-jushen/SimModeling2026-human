# %% [markdown]
"""
# 蒙特卡洛仿真进阶：公交车等待时间悖论

在上一节中，我们学习了蒙特卡洛方法的基本原理，并通过生日悖论和估算圆周率 $\pi$ 体验了它的威力。现在，我们将把这种强大的随机模拟技术应用到更贴近实际生活的场景中：排队论和交通问题。我们将通过模拟公交车到站和乘客等车的过程，来解决一个看似简单却暗藏玄机的概率问题。

## 1. 问题背景：等公交车的烦恼

假设你每天早上都要去同一个公交车站等车。根据公交公司的时刻表，这路公交车**平均每 10 分钟来一班**。

你每天到达车站的时间是完全随机的（比如你可能 8:02 到，也可能 8:07 到）。

**问题：** 你平均需要等多久才能上车？

**直觉答案：** 既然公交车平均 10 分钟来一班，而我到达的时间是随机的，那我应该平均等 $10 / 2 = 5$ 分钟。

这个直觉答案正确吗？让我们用蒙特卡洛仿真来验证一下！

"""

# %% [markdown]
"""
## 2. 建立模型：指数分布与泊松过程

为了进行仿真，我们需要建立一个数学模型。首先，我们需要模拟公交车到站的时间间隔。在现实生活中，公交车不可能像钟表一样精确地每隔 10 分钟准时到达。有时会因为堵车晚点，有时会因为路况好早到。在排队论中，这种随机的时间间隔通常用**指数分布 (Exponential Distribution)** 来描述。

**指数分布**常用于描述独立随机事件发生的时间间隔，例如：
*   两次公交车到站的时间间隔
*   两次电话呼入的时间间隔
*   放射性元素的衰变时间

指数分布的概率密度函数为：
$$f(x; \lambda) = \begin{cases} \lambda e^{-\lambda x} & x \ge 0 \\ 0 & x < 0 \end{cases}$$

其中，$\lambda$ 是率参数（单位时间内事件发生的平均次数）。
指数分布的**均值（期望值）**为 $\mu = 1 / \lambda$。

在我们的问题中，公交车平均每 10 分钟来一班，所以均值 $\mu = 10$ 分钟。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. 模拟公交车到站时间间隔
# 假设我们模拟 10000 辆公交车的到站情况
num_buses = 10000
mean_interval = 10  # 平均间隔 10 分钟

# 使用 np.random.exponential 生成符合指数分布的随机数
# scale 参数即为均值 (1/lambda)
intervals = np.random.exponential(scale=mean_interval, size=num_buses)

# 验证生成的间隔的平均值是否接近 10
print(f"模拟生成的公交车平均间隔时间: {np.mean(intervals):.2f} 分钟")

# 绘制公交车到站间隔时间的直方图
plt.figure(figsize=(8, 5))
plt.hist(intervals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# 绘制理论的指数分布曲线进行对比
x = np.linspace(0, max(intervals), 100)
y = (1/mean_interval) * np.exp(-x/mean_interval)
plt.plot(x, y, 'r-', lw=2, label='Theoretical Exponential PDF')

plt.title("Distribution of Bus Arrival Intervals")
plt.xlabel("Interval (minutes)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()




# %% [markdown]
"""
> **观察图表**：从上面的指数分布直方图中可以看出，虽然平均间隔是 10 分钟，但其实有很多间隔非常短（比如连续来两辆车），也有少数间隔非常长（比如等了半小时都不来）。这就是指数分布常被提到的“长尾”特点。

现在我们有了公交车到站的时间间隔，接下来我们需要模拟乘客（也就是你）随机到达车站的过程，并计算每次的等待时间。我们将模拟成千上万次你到达车站的情景，然后求出最终极的平均等待时间。

## 3. 蒙特卡洛仿真：计算平均等待时间

**代码实现的仿真步骤：**
1.  **生成公交车时刻表：** 我们不能只用相对“间隔”，需要使用累积求和（`np.cumsum`）将间隔时间连续累加起来，得到每辆公交车具体的绝对到站时刻表。
2.  **模拟乘客到达：** 利用均匀分布生成的随机数，模拟乘客到达车站的随机时间点。我们要保证在这个时间轴（第一辆车和最后一辆车之间）内任取一个点。
3.  **计算等待时间：** 对于乘客到达的每一个时间点，立刻向未来“展望”，找到**下一辆**即将到站的首个公交时间。两者相减，就是这次抽样的等待代价。
4.  **统计结果：** 将这个判定重复 10000 次，汇总所有代价，求均值。

"""

# %%
# 1. 生成公交车具体的到站时间点 (绝对时刻表)
# np.cumsum 计算累积和：[a, b, c] -> [a, a+b, a+b+c]
# 假设第一辆车在时刻 0 到站，后续车辆按 intervals 累加
bus_arrival_times = np.cumsum(intervals)

# 2. 模拟乘客到达
num_passengers = 10000
# 乘客到达的时间在第一辆车和最后一辆车之间均匀分布
# np.random.uniform(low, high, size)
passenger_arrival_times = np.random.uniform(
    low=bus_arrival_times[0], 
    high=bus_arrival_times[-1], 
    size=num_passengers
)

# 3. 计算每个乘客的等待时间
wait_times = []

# 遍历每一个乘客的到达时间
for p_time in passenger_arrival_times:
    # 找到所有在乘客到达之后才到站的公交车时间
    # bus_arrival_times > p_time 返回一个布尔数组 [False, False, True, True...]
    future_buses = bus_arrival_times[bus_arrival_times > p_time]
    
    if len(future_buses) > 0:
        # 下一辆车就是 future_buses 中的第一个元素
        next_bus_time = future_buses[0]
        # 等待时间 = 下一辆车到站时间 - 乘客到达时间
        wait_time = next_bus_time - p_time
        wait_times.append(wait_time)

# 4. 统计结果
average_wait_time = np.mean(wait_times)

print(f"乘客平均等待时间: {average_wait_time:.2f} 分钟")
print(f"直觉答案: {mean_interval / 2:.2f} 分钟")

# 绘制等待时间的直方图
plt.figure(figsize=(8, 5))
plt.hist(wait_times, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(average_wait_time, color='red', linestyle='dashed', linewidth=2, label=f'Mean Wait: {average_wait_time:.2f} min')
plt.title("Distribution of Passenger Wait Times")
plt.xlabel("Wait Time (minutes)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()




# %% [markdown]
"""
## 4. 揭秘等待时间悖论

仿真结果出乎意料！我们的蒙特卡洛统计得出的平均等待时间竟然接近 10 分钟，而不是直觉上所认为的 5 分钟。这在排队论里被称为**等待时间悖论 (Waiting Time Paradox)** 或**检查悖论 (Inspection Paradox)**。

为什么会这样？蒙特卡洛仿真不仅排除了我们的直觉错误，还能帮助我们直观地理解背后的原因。

**原因在于著名的“采样偏差”现象：**
虽然公交车的平均间隔是 10 分钟，但间隔时间有长有短（如我们之前绘制的指数分布图所示）。
当你**随机**到达车站时，你更有可能落入一个**较长**的时间间隔内，而不是一个较短的时间间隔内。

想象一下：
*   间隔 A：2 分钟
*   间隔 B：18 分钟
平均间隔是 $(2+18)/2 = 10$ 分钟。
如果你随机到达，你落在间隔 B（18分钟）内的概率是落在间隔 A（2分钟）内的 **9 倍**！
一旦你落入间隔 B，你的平均等待时间就是 $18 / 2 = 9$ 分钟。

因此，较长的间隔在计算乘客平均等待时间时占据了更大的权重，拉高了整体的平均值。在指数分布的特殊情况下，这个平均等待时间恰好等于平均间隔时间本身（10分钟）。

**总结：**
蒙特卡洛仿真不仅是一个强大的计算工具，更是一个**验证直觉、发现反直觉现象**的利器。当面对复杂的概率或系统问题时，写一段简单的仿真代码，往往比绞尽脑汁推导公式更有效、更直观。

"""
