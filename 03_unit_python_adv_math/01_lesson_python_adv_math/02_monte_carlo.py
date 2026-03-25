# %% [markdown]
"""
# 蒙特卡洛仿真 (Monte Carlo Simulation)

## 1. 什么是蒙特卡洛方法？

除了前面学习的微分方程外，**蒙特卡洛（Monte Carlo）方法**是另一种极其重要的数值计算技术。**它不依赖于严密的解析推导，而是通过大量的随机抽样来估算数学问题的结果。**这种方法在处理高维积分、复杂概率问题和系统风险评估时非常有效。

蒙特卡洛是摩纳哥的一个著名赌城。赌博本质上是一种基于概率和随机性的游戏。蒙特卡洛方法正是借用了这个名字，它是一种**基于随机抽样来解决计算问题**的算法。

为什么“随机”能解决“确定性”的数学问题？
这依赖于概率论中的**大数定律 (Law of Large Numbers)**：只要我们进行的随机试验次数足够多，样本的平均值就会无限接近于真实的数学期望值。

![Monte Carlo Pi Estimation](https://upload.wikimedia.org/wikipedia/commons/4/40/Circle_area_Monte_Carlo_integration.png)
*(图示：通过在正方形内随机撒点，统计落在内切圆内的点的比例，可以估算出圆周率 $\pi$ 的值)*

"""

# %% [markdown]
"""
## 2. 实例 1：生日悖论 (Birthday Paradox)

> **问题**：假设一个房间里有 $n=50$ 个人，求至少有 2 人生日相同的概率。

对于这个经典问题，用传统概率论直接解析推导虽然可行，但如果规则稍微变得复杂一点（比如考虑闰年、双胞胎、或者是特定节假日出生率波动），解析计算的难度就会呈指数级爆炸增长。

而这正是**蒙特卡洛方法大显身手**的地方。我们退一步海阔天空，不需要推导长长的复杂公式，只需要写几行简单的代码，让计算机疯狂模拟成千上万次“随机生成 50 个人生日”的过程，然后“简单粗暴”地统计出现不同由于的碰撞次数比例即可！

**理论计算方法：**
直接计算“至少有两人相同”很复杂，我们通常计算它的反面：“所有人生日都不同”的概率 $P_{diff}$。
第一个人有 365 种选择，第二个人为了不重复只有 364 种选择，以此类推...
$$P_{diff} = 1 \times \frac{364}{365} \times \frac{363}{365} \times \dots \times \frac{365-n+1}{365}$$
那么至少两人相同的概率就是 $1 - P_{diff}$。

**蒙特卡洛仿真方法：**
1.  随机生成 50 个 1 到 365 之间的整数，代表 50 个人的生日。
2.  检查这 50 个数字中是否有重复的。如果有，记为一次“命中 (Hit)”。
3.  重复上述过程 1000 次（或更多）。
4.  最终概率 $\approx$ 命中次数 / 总模拟次数。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
n_people = 50        # 房间里的人数
iterations = 2000    # 模拟的总次数

# 记录每次模拟后计算出的当前概率估计值，用于画图
probability_estimates = []
hits = 0 # 记录出现重复生日的次数

# 1. 蒙特卡洛模拟过程
for i in range(1, iterations + 1):
    # 随机生成 n_people 个生日 (1 到 365 之间的整数)
    # np.random.randint 是左闭右开区间，所以上限写 366
    birthdays = np.random.randint(1, 366, n_people)
    
    # 使用 np.unique 获取不重复的元素
    unique_birthdays = np.unique(birthdays)
    
    # 如果不重复的元素个数小于总人数，说明有重复的生日！
    if len(unique_birthdays) < len(birthdays):
        hits += 1
        
    # 记录当前的概率估计值
    probability_estimates.append(hits / i)

# 2. 理论值计算
# np.arange(365, 365-n_people, -1) 生成 [365, 364, ..., 365-n+1]
a = np.arange(365, 365 - n_people, -1) / 365.0
ideal_prob = 1 - np.prod(a)

print(f"理论概率值: {ideal_prob:.4f}")
print(f"蒙特卡洛模拟 {iterations} 次后的估计值: {probability_estimates[-1]:.4f}")

# 3. 绘制估计值随模拟次数变化的收敛曲线
plt.figure(figsize=(10, 5))
plt.plot(probability_estimates, label='Monte Carlo Estimate')
plt.axhline(ideal_prob, color='red', linestyle='--', label='Theoretical Value')
plt.title("Monte Carlo Simulation of Birthday Paradox (n=50)")
plt.xlabel("Number of Iterations")
plt.ylabel("Estimated Probability")
plt.legend()
plt.grid(True)
plt.show()

# 你可以看到，随着模拟次数的增加，蓝色的估计曲线逐渐稳定并收敛于红色的理论值。




# %% [markdown]
"""
## 3. 实例 2：用蒙特卡洛方法估算 $\pi$

蒙特卡洛方法不仅能算概率，也是对付**几何求面积和不规则曲线微积分**的利器。其中最著名的科普例子就是通过“随手撒豆子”来估算圆周率 $\pi$。

通过这种“随机撒点+统计符合条件的比例”的方式，我们将一个纯粹的“几何面积问题”，极为巧妙地转化为了一个“概率分布计算问题”。

**原理：**
1.  假设有一个边长为 2 的正方形，其面积为 $2 \times 2 = 4$。
2.  在正方形内画一个内切圆，半径 $r = 1$，其面积为 $\pi r^2 = \pi$。
3.  我们在正方形内**均匀地随机撒点**。
4.  一个点落在圆内的概率 $P = \frac{\text{圆的面积}}{\text{正方形的面积}} = \frac{\pi}{4}$。
5.  因此，$\pi \approx 4 \times \frac{\text{落在圆内的点数}}{\text{总撒点数}}$。

"""

# %%
# 撒点总数
num_points = 10000

# 1. 在 [-1, 1] 范围内随机生成 x 和 y 坐标
# np.random.uniform 生成均匀分布的随机数
x_points = np.random.uniform(-1, 1, num_points)
y_points = np.random.uniform(-1, 1, num_points)

# 2. 计算每个点到原点 (0,0) 的距离的平方
distances_squared = x_points**2 + y_points**2

# 3. 判断点是否在圆内 (距离平方 <= 1)
# inside_circle 是一个布尔数组 (True/False)
inside_circle = distances_squared <= 1

# 4. 统计圆内的点数
points_in_circle = np.sum(inside_circle)

# 5. 估算 pi
pi_estimate = 4 * points_in_circle / num_points

# 计算收敛过程：前 n 个点中落在圆内的点数比例 * 4
# 注意 np.arange(1, num_points+1) 是为了防止除以 0
cumulative_pi = 4 * np.cumsum(inside_circle) / np.arange(1, num_points + 1)

print(f"撒点总数: {num_points}")
print(f"落在圆内的点数: {points_in_circle}")
print(f"估算的 Pi 值: {pi_estimate:.5f}")
print(f"真实的 Pi 值: {np.pi:.5f}")

# 6. 可视化
plt.figure(figsize=(12, 5))

# 图1：撒点展示 (为了图表清晰，我们只画前 2000 个点)
plt.subplot(1, 2, 1)
plot_limit = min(2000, num_points)

# 画出圆内的点 (蓝色)
plt.scatter(x_points[:plot_limit][inside_circle[:plot_limit]], 
            y_points[:plot_limit][inside_circle[:plot_limit]], 
            color='blue', s=5, label='Inside Circle')

# 画出圆外的点 (红色)
plt.scatter(x_points[:plot_limit][~inside_circle[:plot_limit]], 
            y_points[:plot_limit][~inside_circle[:plot_limit]], 
            color='red', s=5, label='Outside Circle')

# 画一个标准的圆圈作为参考
circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title(f"Monte Carlo Pi Estimation (N={plot_limit})")
plt.legend(loc='upper right')
plt.gca().set_aspect('equal', adjustable='box') # 保证 x 和 y 轴比例一致，圆才不会变扁

# 图2：大数定律收敛过程
plt.subplot(1, 2, 2)
plt.plot(cumulative_pi, color='blue', alpha=0.8, label='Estimated $\pi$')
plt.axhline(np.pi, color='red', linestyle='--', label='True $\pi$')
plt.title("Convergence of $\pi$ Estimate (Law of Large Numbers)")
plt.xlabel("Number of Trials (N)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




# %% [markdown]
"""
## 练习：蒙特卡洛计算抛物线面积（定积分）

你已经见识过怎么用蒙特卡洛计算复杂的 $\pi$ 了，实际上这个撒点法同样极其适合用来求解**数值面积（定积分）**。

请补全代码框中的 TODO 任务，求函数：
$f(x) = x^2$ 
在区间 $x \in [0, 1]$ 时，下方围成的面积。

"""

# %%
import numpy as np

# 任务：利用蒙特卡洛方法，估算函数 f(x) = x^2 在区间 [0, 1] 下方的面积 (即定积分)
# 这个区域可以被包含在一个 x 在 [0, 1], y 在 [0, 1] 的边长为 1 的正方形内

num_points = 100000

# TODO 1: 在 [0, 1] 的正方形空间内随机撒 100000 个 (x, y) 坐标
x_rand = 
y_rand = 

# TODO 2: 判断这些点有哪些落在了曲线 y = x^2 的下方 (即对每个点满足 y < x^2)
under_curve = 

# TODO 3: 统计符合条件的点数，除以总点数，乘以正方形总面积(1x1=1)，即可估算积分面积
area_estimate = 

# 理论值我们大家用微积分应该算得出：对于 x^2 在 [0, 1] 的积分等于 1/3
# print(f"你的蒙特卡洛估算面积: {area_estimate:.4f}")
# print(f"真实的面积积分理论值: {1/3:.4f}")
