# %% [markdown]
"""
# 5. 非线性方程与方程组求解：寻找平衡的艺术

在线性系统的世界里（如纯电阻网络），我们可以精确地构建矩阵并瞬间求出唯一解。
但现实世界往往是**非线性**的。比如，晶体管的电流-电压曲线是指数型的，风阻与速度的平方成正比。

一旦方程里出现了 $\sin(x)$, $e^x$, $x^2$ 甚至它们的组合，就没有通用的公式解法了。
这时候，我们必须使用**数值优化算法**，让计算机盲猜一个值，然后根据误差反向调整，一步步“逼近”那个让方程成立的根。

SciPy 提供的一大神器是 `scipy.optimize.fsolve`。

![Nonlinear Roots](https:///upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Newton_iteration.svg/300px-Newton_iteration.svg.png)
*(非线性方程的求解，在几何上相当于找出曲线穿过 $y=0$ 轴的交点，算法常常像牛顿法一样沿着切线“滑”至坑底寻找根。)*

"""

# %% [markdown]
"""
## 求解单个非线性方程

假设你在设计一个斜屋顶，需要找到一个支撑角度 $x$（弧度），满足极其复杂的几何约束方程式：
$\tan(x) = \sqrt{\left(\frac{8}{x}\right)^2 - 1}$

第一步也是最核心的一步：把所有项移到等号一侧，把它变成 **$f(x) = 0$** 的形式。
让 $f(x) = \tan(x) - \sqrt{\left(\frac{8}{x}\right)^2 - 1}$

第二步：提供一个**初始猜测值 (Initial Guess)**，因为非线性方程经常有多个解（多个波峰波谷穿过横轴）。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. 定义我们转换好的目标函数 f(x)
def f(x):
    return np.tan(x) - np.sqrt((8.0/x)**2 - 1.0)

# 2. 从猜测的角度 x = 1.0 弧度开始尝试寻找根
guess_1 = 1.0
root1 = fsolve(f, guess_1)

print(f"从 {guess_1} 开始搜寻，找到的根是 x = {root1[0]:.4f}")
print(f"验算: f(x) 的值是否趋近于0? {f(root1[0])}")

# 3. 如果我们猜从 x = 4.5 开始呢？
guess_2 = 4.5
root2 = fsolve(f, guess_2)
print(f"\n从 {guess_2} 开始搜寻，找到的根是 x = {root2[0]:.4f}")




# %% [markdown]
"""
### 为什么“初始猜测值”比算法更重要？（可视化分析）

为什么 `fsolve` 刚才输出了两个截然不同的解？
非线性方程就像群山，`fsolve` 就像一个被蒙上眼睛的人，他只会顺着当前斜坡往下走到最近的“坑底”（数学上的局部最优解/局部根）。

解决非线性问题的标准工程师流程：**先画图，再盲猜，最后计算机精确求解。**

"""

# %%
# 我们把刚才那个奇怪函数画出来看看
xs = np.linspace(1, 8, 400)
# 规避某些极点处会报警告
ys = f(xs)

plt.figure(figsize=(10, 5))
plt.plot(xs, ys, label='$f(x) = \tan(x) - ...$')

# 画一条红色的横线代表 y=0
plt.axhline(0, color='red', linestyle='--', label='y=0 Goal')

# 把 fsolve 刚才找到的两个点标记上去
plt.scatter([root1[0], root2[0]], [0, 0], color='green', s=100, zorder=5, label='fsolve Roots')

plt.ylim(-10, 10)
plt.title("可视化诊断：为什么会有多个解")
plt.xlabel("角度 x (rad)")
plt.ylabel("误差 f(x)")
plt.legend()
plt.grid(True)
plt.show()

# 很明显，由于 tan(x) 的周期性波动，这条曲线多次穿过了红色的零线。
# 所以，猜测值的设置决定了 fsolve 会“滚”向左边还是右边的交点！




# %% [markdown]
"""
## 求解非线性方程“组” (多个未知数互相纠缠)

工业界的真实挑战通常是多个参量必须同时满足一组非线性方程。
假设你正在分析一个水管分流系统，得到了两个纠缠在一起的方程：
$$ y - x^2 = 7 - 5x $$
$$ 4y - 8x = -21 $$

思想同上，我们要构建一个**向量输入**到**向量输出**的函数。
1. 让输入为 `vars = [x, y]`
2. 将方程全部移动至一侧，写成 $F(x, y) = 0$ 格式
3. 提供一组包含 `x, y` 两个元素的猜测点。

"""

# %%
from scipy.optimize import fsolve

# 定义系统方程组
# 输入 vars 是一个数组，包含了我们要找的所有未知数
def pipe_system(vars):
    # 解包出未知数
    x = vars[0]
    y = vars[1]
    
    # 构建所有 f(x, y) = 0 形式的方程式
    eq1 = (y - x**2) - (7 - 5*x)
    eq2 = (4*y - 8*x) - (-21)
    
    # 把这些误差值打包成列表返回
    return [eq1, eq2]

# 提供一个二维的猜测点 (x=4, y=5)
# (在没有任何头绪时，常常随便乱猜一个起点，例如全0或全1)
initial_guesses = [4.0, 5.0]

solution = fsolve(pipe_system, initial_guesses)
x_sol, y_sol = solution

print(f"方程组的非线性根为: x = {x_sol:.4f}, y = {y_sol:.4f}")

# 验算：把求出解代回原函数，看误差是不是极其接近 [0, 0]
print("极小的残差误差:", pipe_system(solution))




# %% [markdown]
"""
### 自己动手：求解混合非线性方程组

在研究气态燃烧物反应时，你列出了关于生成物浓度 $C_1$ 和压力 $P_2$ 的复杂方程组：
1. $C_1^2 + P_2^2 - 25 = 0$  (能量/物质守恒，实际上是一个圆)
2. $C_1 \cdot P_2 - 12 = 0$ (状态方程，实际上是一条双曲线)

请使用 `fsolve` 求解这个方程组。已知根据物理意义，$C_1$ 和 $P_2$ 都必须是大于 0 的实数。
因此请你**合理选择你的初始猜测值 `initial_guess` 均为正数**（例如都从 2.0 开始）。

在下方写出代码，并将求解结果分别赋给变量 `ans_c1` 和 `ans_p2`。

"""

# %%
from scipy.optimize import fsolve

# 1. 定义反应系统方程
def reaction_sys(vars):
    C1 = vars[0]
    P2 = vars[1]
    
    # 请填入方程 1:  C1^2 + P2^2 - 25
    eq1 = # TODO
    # 请填入方程 2:  C1 * P2 - 12
    eq2 = # TODO
    
    return [eq1, eq2]

# 2. 设置符合物理意义（全为正数）的初始猜测点
initial_guess = [2.0, 2.0]

# 3. 求解
solution = # TODO
ans_c1, ans_p2 = solution

print(f"物料浓度 C1: {ans_c1:.4f}, 压力 P2: {ans_p2:.4f}")
