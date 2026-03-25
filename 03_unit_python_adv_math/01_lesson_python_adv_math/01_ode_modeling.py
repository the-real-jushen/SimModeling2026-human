# %% [markdown]
"""
# 高级数学计算：常微分方程 (ODE) 建模与求解

## 1. 为什么要学习微分方程？

因为整个宇宙中的一切基本都可以用微分方程来建模！

从人口数量的增长模型、流体力学中的纳维-斯托克斯方程，到电磁学中最著名的麦克斯韦方程组，微分方程无处不在。

最基础也最实用的微分方程，就是牛顿第二定律：$F = ma$。
加速度 $a$ 是位移 $x$ 对时间 $t$ 的二阶导数，所以这个公式本质上是一个微分方程：

$$
F = m \frac{d^2x}{dt^2}
$$

为什么微分方程如此重要？因为我们面对的现实世界是**动态变化**的，而不是静止的。要描述一个变化的系统，我们需要描述它的**变化规律**，而“导数”或“微分”正是描述变量变化率的数学工具。

![Lorenz Attractor](https://upload.wikimedia.org/wikipedia/commons/b/bb/Lorenz_Attractor_from_Gauss-Legendre.png)
*(图示：洛伦兹吸引子，一个由三个非线性常微分方程描述的混沌系统，展示了相空间中的复杂轨迹)*

"""

# %% [markdown]
"""
## 2. 什么是微分方程？

简单来说，微分方程就是包含未知函数及其导数的方程。

一般形式可以写成状态变量的变化率依赖于时间和当前状态：

$$
\frac{dy}{dt} = f(t, y)
$$

我们的目标是解出背后的运行轨迹，也就是求出这个隐藏的函数 $y = g(t)$。

*   **常微分方程 (ODE - Ordinary Differential Equation)**: 只有一个自变量（例如时间 $t$）。这是我们本节课的重点。
*   **偏微分方程 (PDE - Partial Differential Equation)**: 有多个自变量（例如时间 $t$ 和空间坐标 $x, y, z$）。PDE 的求解要复杂得多，通常需要使用有限元分析等高级方法。

### 数值求解的必要性

对于简单的受恒力推导，我们可以轻易求出解析解：

$$
x(t) = \frac{1}{2} \frac{F}{m} t^2
$$

但如果物理力 $F$ 是随时间、系统速度甚至位置不断变化的极度复杂函数呢？这在空气阻力、磁场引力中非常常见。此时解析解（也就是大家在微积分课上背的公式）往往根本不存在。所以我们必须依靠计算机，利用**数值积分**的方法，在极微小的时间步长 $\Delta t$ 里计算导数，像走台阶一样一步一步地递推出系统在随后各个时间点的状态。

"""

# %% [markdown]
"""
## 3. 微分方程建模流程：追踪曲线问题

为了让计算机帮你解复杂的微分方程，你个人的首要任务，是把物理问题转化为**一阶微分方程组的标准形式**。这需要以下核心建模流程：
1.  **确定未知函数与状态向量**：我们要预测什么？比如二维平面上的坐标，我们把它打包成向量 $\vec{Y}(t) = [x(t), y(t)]$。
2.  **寻找导数/变化率关系**：利用物理规律，写出每一个状态变量在此刻的导数 $\left[ \frac{dx}{dt}, \frac{dy}{dt} \right]$ 如何由当前系统里的信息计算推导出来。
3.  **确定初始条件**：此时处于 $t=0$ 时的宇宙起点状态是什么？（这被称为初值 IVP）。

### 经典案例：导弹追踪敌舰 (Pursuit Curve)

> **问题描述**：
> 设位于坐标原点 $(0,0)$ 的甲舰向位于 $x$ 轴上点 $A(1,0)$ 处的乙舰发射导弹。导弹头**始终对准**乙舰。
> 乙舰以速度 $v_s = 1$ 的巡航速度沿平行于 $y$ 轴的直线匀速向上逃窜，导弹的速度恒为 $v_m = 5$。
> 要求写出导弹坐标随时间的运行轨迹，以及导弹将在何时何地击中乙舰？

**建模推导过程：**

1.  **目标向量（要预测的状态）**：
    我们将导弹此刻在平面坐标系所处的位置定义为待求解的数组 $\vec{Y}(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix}$。

2.  **已知条件运算**：
    *   乙舰因为是匀速单向直线运动，所以任意时刻的绝对坐标很容易直接写出为 $S(t) = (1, t)$ 。
    *   导弹速度的大小是定死的： $|\vec{V}_m| = 5$。
    *   导弹最核心的追踪逻辑是：“弹头永远对准敌舰”。这意味着导弹的**速度方向向量**就是此刻导弹坐标指向敌舰坐标的向量。

3.  **列写微分方程**：
    我们先写出从导弹指向敌舰的真实相对位置差向量：

    $$
    \vec{D} = S(t) - M(t) = \begin{bmatrix} 1 - x \\ t - y \end{bmatrix}
    $$
    
    导弹与敌舰的绝对距离（也就是长度分母）为：

    $$
    L = \sqrt{(1-x)^2 + (t-y)^2}
    $$

    然后我们将指向向量 $\vec{D}$ 除以本身的长度 $L$ ，这样就得到里一个长度恰好为 1，但是指着敌人的“**单位方向向量**”。
    
    最后，因为导弹真实速度是 5，所以只要用 5 乘以这个单位方向向量，就获得了导弹当前在这个方向上的 $X$-分量速度和 $Y$-分量速度：

    $$
    \begin{bmatrix} \frac{dx}{dt} \\ \frac{dy}{dt} \end{bmatrix} = 5 \cdot \frac{1}{L} \begin{bmatrix} 1 - x \\ t - y \end{bmatrix}
    $$

    这就是我们需要的标准的、直接定义每个导数状态的**一阶微分方程组**！其数学结构已无懈可击。

4.  **初始条件 IVP**：导弹从原点满弹待发，所以毫无疑问：$\vec{Y}(0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。

"""

# %% [markdown]
"""
## 4. 使用 Python 求解 ODE

我们将使用 `scipy.integrate.solve_ivp` (Solve Initial Value Problem，解初值问题) 来跑上面的方程极值。

**Python 函数结构要求：**
不论你的数学微分公式长什么样，你必须在 Python 中构造一个只接受 `(t, y)` 两个参数的函数（其中 `y` 传进来是一个数组）。
并且函数最终必须 `return` 返回一个装着各个状态量速度/导数的计算结果数组 $[ \frac{dx}{dt}, \frac{dy}{dt} ]$。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 把纯粹的物理方程写成标准 Python 函数 (必须接收 t 还有状态数组 y)
def pursuit_eq(t, y):
    # 第一步往往是从 y 数组里面分别解包出当前各变量的名字，防止混淆代码
    mx = y[0] # 当前时刻导弹的 x 位置
    my = y[1] # 当前时刻导弹的 y 位置
    
    # 计算当前两舰之间两点距离分母 (根据勾股定理)
    dist = np.sqrt((1 - mx)**2 + (t - my)**2)
    
    # 增加安全性（避免快撞上时，除以 0 导致计算机报错崩溃）
    if dist < 1e-8:
        return [0, 0]
        
    # 根据我们数学建模推导得出的方程，计算分量的导数 (速度)
    dx_dt = 5 * (1 - mx) / dist
    dy_dt = 5 * (t - my) / dist
    
    # 要求必须返回包含他们俩导数的结构：[dx/dt, dy/dt]
    return [dx_dt, dy_dt]

# 2. 设置环境参数与目标
t_span = [0, 0.3]  # 规定时间推演跨度的框架区间 (这里是 t=0 秒 到 0.3 秒)
y0 = [0, 0]        # 给定最初 t=0 时的系统状态 -> [导弹位于x=0, 位于y=0]

# 3. 呼叫引擎求解
# max_step 限制步长上限以保证曲线记录细腻（如果不设置它可能走步太大导致轨迹僵硬和越界差错）
sol = solve_ivp(pursuit_eq, t_span, y0, max_step=0.005)

# 4. 提取和绘图
# sol 是系统抛回来的结果大对象：
# sol.t 存放了当时一步步行走的全部真实时间点
# sol.y 存着对应点的全部位置 [存着所有 x 记录, 存着所有 y 记录]
missile_x = sol.y[0]
missile_y = sol.y[1]

plt.figure(figsize=(8, 6))
# 勾勒出敌舰笔直逃跑的目标航线 (在 x=1 位置画虚线代表一直往上开)
plt.axvline(x=1, color='red', linestyle='--', label='Target Path')
# 画出我们解出的巡航导弹的动态跟踪曲线
plt.plot(missile_x, missile_y, 'b-', linewidth=2, label='Missile Trajectory')

plt.title("Missile Pursuit Curve (Target vs Missile)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
"""
## 5. ODE 求解中的高级探测：“事件” (Events)

在上面的基础求解中，你可能已经发现了一个致命的问题：**盲目性时间限**。
我们硬生生在脑子里估计设定了 `t_span = [0, 0.3]` 作为结束判定。但如果你正在开发一个工业自动雷达或者计算弹道截击，你怎么可能未卜先知这颗导弹它什么时候炸（碰到目标）？如果你推移的时间短了，没撞上；推移的时间长了，导弹会穿过敌舰继续狂飞乱跑（因为没有条件让它停下）。

这时候就需要引入牛逼的 **“事件机制” (Events)**。

### 什么是事件机制的数学原理？
你可以向系统传递自定义的观察哨函数（类似于监听器）。这就好像你在时间推移的赛道旁边埋了一根**红外绊线**。
事件函数的唯一任务，就是计算并返回一个**监控值**。`solve_ivp` 在推算每一时间步的同时，都会去检查你给的这个值。
当观察到这个特征值**刚好穿过数字 0（也就是正负发生翻转跨越时）**，求解器内部精密算法就会立刻向后退几步，并使用复杂的插值精确卡住锁定刚才产生“零点切割”时那一极微小精确的瞬间！

**举个通俗的例子**：你扔一个球抛物落下，你不知道什么它什么时候砸到地板（即 $y = 0$）算结束。
那么你立刻定义：`return current_y_height`，告诉积分器：“由于你推算过程中球的高度 `y` 一直在从正数变小小小小，终于在某一步球踩出了地面陷进泥里导致 `y` 变成负数（负标高），这说明在刚刚极短的时间里经过了跨度 0，事件触发拉响警报！记录那一秒的时间点！”

我们还可以给事件附带极其强大的属性：
*   **`terminal = True`**：这意味着此事件为发生即“终结性死亡打击”，积分模拟立刻在此刻彻底停止不再计算接下来没意义的时光（就像导弹已炸）。
*   **`direction = -1`**：它意味着必须强控制是从正数降低跌落成负数，跨过0时才响警报！（避免从负面上升穿出时误报）。

"""

# %%
# 定义事件 1：导弹击中敌舰
# 这是终止条件：我们把导弹和敌舰实时算出的实际距离和容忍爆炸误差做个减法差值
def hit_event(t, y):
    mx, my = y[0], y[1]
    # 计算当前的瞬间实时绝对距离
    dist = np.sqrt((1 - mx)**2 + (t - my)**2)
    # 当你们实时距离被不断缩减，减到了 0.005 以下的贴边爆炸范围时，整体值就会由正降为负，触发 0 取点事件
    return dist - 0.005

# 打上终止标志，击中即刻触发死亡终止推算
hit_event.terminal = True

# 定义事件 2：导弹进入防空火力警报范围
def in_range_event(t, y):
    mx, my = y[0], y[1]
    dist = np.sqrt((1 - mx)**2 + (t - my)**2)
    # 只要进入雷达 0.5 半径就记录下来警报器响时刻
    return dist - 0.5

# 我们要求该警报只在首次从“距离远大于 0.5”（正值计算结果）突变为“进入小于 0.5”（负值）时鸣笛
in_range_event.direction = -1


# 这回有恃无恐不管时间了，直接把求解最大时间放得非常漫长 (t=2)，反正有 terminal 打断保底
t_span_long = [0, 2.0]
# 将我们刚写的哨兵事件塞进 events=[] 里放生
sol_events = solve_ivp(pursuit_eq, t_span_long, y0, 
                       events=[hit_event, in_range_event], 
                       max_step=0.005)


# 开始绘图表现系统触发战果
plt.figure(figsize=(10, 6))
plt.axvline(x=1, color='red', linestyle='--', label='Target Cruiser Path (x=1)')
plt.plot(sol_events.y[0], sol_events.y[1], 'b-', label='Missile Precision Trajectory')

# sol_events.y_events 和 t_events 中包含了各个哨兵事件被拦截触发时记录下的坐标和时间点情况！
if len(sol_events.y_events[1]) > 0: # 如果防空雷达成功警报
    range_x = sol_events.y_events[1][0][0]
    range_y = sol_events.y_events[1][0][1]
    plt.scatter(range_x, range_y, color='orange', marker='^', s=120, label='Entered 0.5 Danger Zone', zorder=5)

if len(sol_events.y_events[0]) > 0: # 如果被记录下了终结爆炸瞬间
    hit_x = sol_events.y_events[0][0][0]
    hit_y = sol_events.y_events[0][0][1]
    hit_time = sol_events.t_events[0][0]
    plt.scatter(hit_x, hit_y, color='red', marker='*', s=300, label=f'HIT Target Detonation at t={hit_time:.3f}s', zorder=5)

plt.title("Pursuit Curve Trajectory (Controlled via Autonomous Events)")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()




# %% [markdown]
"""
让我们来测试一下你对高阶微分方程降阶基础思路的理解。

"""

# %% [markdown]
"""
## 6. 重要：利用变量降阶解决更高阶微分方程 (二阶 ODE)

刚刚测验里点出了一个核心痛点：所有的物理规律基本都是通过牛顿第二定律 $F = ma$ 建立的。
而加速度 $a$ 本身是物体位置 $y$ 随着时间求了**两次导数的结果（即 $a = y''$）。它是一个二阶常微分方程！**

但是 `solve_ivp` 和几乎所有的数值积分器都**只能处理一层状态的“第一阶”导数（仅仅只能处在速度层面 $\vec{Y}' = \dots$）。也就是它不可能看懂 $y''$ 这个指令。**
这就需要我们用到一个极为精妙的数学降维技巧：**将单一高阶转化解构成包含多元信息的一阶矩阵包！** 

### 举例：在地球重力下的上抛小球运动 (只有单一常数力)

已知小球的位置 $y$ 只受到向下一直固定的重力加速度影响：$-9.8$。
二阶物理方程写下便是：
$$
\frac{d^2y}{dt^2} = -9.8 
$$

**降阶解耦大法开始：**
1. 无中生有：我们凭空定义一个极其自然的新名词“速度”变量叫做 $v$。
   我们宣称它就等价于原来 $y$ 的第一阶导数：

   $$
   v = \frac{dy}{dt}
   $$

2. 换名术：既然 $v$ 代表着一阶导数，那原来棘手的极其恐怖的那个原来的二阶项（对 $v$ 再求一次导的家伙）自然就被替代更替成了：。

   $$
   \frac{dv}{dt} = -9.8
   $$

3. 我们把方程揉捏组装在一个包含高度位置和物理速度的大向量里（即我们让状态数组 $\vec{Y} = [y, v]$ 包揽所有数据）：

   $$
   \frac{d}{dt}\begin{bmatrix} y \\ v \end{bmatrix} = \begin{bmatrix} v \\ -9.8 \end{bmatrix}
   $$

此时，在这个最新的微分框架矩阵下，整个函数里**只含有最高“一阶”导数结构**的数学算式！
下面我们就用代码把这个降维后的一阶数组输入引擎中实际测验一下。

"""

# %%
# 定义符合求解器要求的一阶降阶组合矩阵
def throw_ball(t, Y):
    # 根据我们装进数组时的顺序提取状态：Y[0]代表它目前的物理高度，Y[1]携带着它此刻的当前飞行速度
    y_height = Y[0]
    v_velocity = Y[1]
    
    # 填装矩阵要求：[dy/dt 的结果, dv/dt 的结果]
    # 我们刚刚从数学降阶推演知：
    # 1. 高度的导数（也就是变化率）不就刚好等于我们定义的 v_velocity 吗？
    # 2. 而速度的导数（也就是加速度）此时则受唯一的地球常数即 -9.8 所控制！
    return [v_velocity, -9.8]


# 定义一个终止落地的观测事件 (Event)：小球什么砸到地板结束演算
def hit_ground_event(t, Y):
    return Y[0] # 检测当这高度 y 值落至碰到 0 这个地板数值点时翻转并触发报警

hit_ground_event.terminal = True
# 极其重要的属性：因为我们就是从高度 0 往上起抛的，如果不管方向，刚开始 t=0 时高度为 0 就会直接触发终止！
# 设定 direction = -1，要求只有在 y 轴高度“从正值下落穿透到零及负值”时，才会被判定为真正砸中地面。
hit_ground_event.direction = -1

# 设置一个极为夸张的时间 (比如长达 10s)，并给予系统大力的向上抛初始力量和初始地面位置 (初值在 0 米高度，赋予初始开出速度以 20m/s 的上冲力量跃起)
sol_throw = solve_ivp(throw_ball, [0, 10.0], [0, 20.0], events=[hit_ground_event], max_step=0.01)

plt.figure(figsize=(8, 4))

# 既然只抛上抛下横轴没实地坐标，那大家就观察时间 t 横轴和小球的高度记录(sol_throw.y[0])：
# 使用带有 marker 的线条，可以清晰看出求解器在时间轴上踩下的每一个"计算脚印"
plt.plot(sol_throw.t, sol_throw.y[0], 'g.-', linewidth=2, markersize=8, label="Simulated positions")
plt.title("Upward Sown Ball Trajectory under Constant Gravity (-9.8)")
plt.xlabel("Time Passed (s)")
plt.ylabel("Height off Ground (m)")

hit_time = sol_throw.t_events[0][0]
plt.scatter(hit_time, 0, color='red', marker='x', s=100, label=f"Hit floor precisely at {hit_time:.2f}s")
plt.legend()
plt.grid(True)
plt.show()

# 到此你应该彻底融会贯通，只要能利用降维矩阵引入速度等中间量，牛顿的一切动态经典物理规律均可利用 solve_ivp 给攻破并推演呈现而出！




# %%




# %% [markdown]
"""
## 综合练习：重力弹击 10 次的能量衰减皮球

现在，让我们把“降维微分算式”、“事件（Events）机制”彻底融会贯通，解决一段极其常见的带有阶段打断性（碰撞）物理建模任务。

**场景描述**：
一个小球从地面被以 $v = 15 \text{m/s}$ 的初速度向上抛出。在此后它只受到地球重力影响。
但现在与上文不同：加入了真实反弹逻辑，**小球每次砸回地面，机器程序不能到此彻底关停。小球会立刻凭借弹力弹跳起飞！但因为由于每次撞击消耗了机械能，弹起时全新的初始速度大小，只有上一秒落地撞击瞬间那个触地速度的 $0.9$ 倍。**

请连续推演它在地上反复横跳 **10 次** 的完整动态人生轨迹！

**编程任务与工程提示**：
1. **循环启动接力机制**：因为触地撞击瞬间出现了**突发的不连续动能损耗（即速度直接被外力强行翻转减弱）**，所以你无论多么厉害，都不可能用 1 轮方程解完全程。你必须要：设定循环 -> 执行 `solve_ivp` -> 被坠地事件拦截叫停 -> **从被拦截的尸体残骸数据里，提取落地时刻与最终速度信息，人工计算并刷新乘以 $0.9$ 后** -> 塞进新的环境包里作为再次启动引擎的新初始条件。
2. **寻找多重事件**：系统能一次性塞入多个监控探头。第一个自不消说，是监控撞击地面的拦截终结者探头。
   此时需要你加装第二个探头去用来记录**最高点**。这非常考验你的高中微积分：到底当什么物理量，发生了过 0 指标时，能完美意味着自己此时此时正悬停且处于最高处呢？（另外别忘了！只用来插眼监控的数据事件，千万不要手滑打上 `terminal=True` 这个断电拦截标签！）
3. 别忘最后把这一整台人生履历通过之前学到的库画进同一张绘图板之中。

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 物理常数与初始状态
g = 9.8
y0_state = [0.0, 15.0] # 初始高度 0m，初始向上起跳速度 15m/s
t_start = 0.0          # 系统时间计数器
bounce_times = 10      # 需要模拟弹跳的总次数

# 准备用于记录拼接长图数据的空列表
all_t = []
all_y = []
apex_t = []
apex_y = []

# TODO 1: 定义重力影响下的速度与高度二阶微分降级后的状态方程
def ball_dynamics(t, Y):
    pass
    
# TODO 2: 定义两个观察事件函数
# 事件A：小球砸倒地板 (判定高度)
def hit_ground(t, Y):
    pass

# 事件B：记录当前波段的最高点！ (想一想，最高点时是哪个物理量发生破 0 翻转？)
def highest_point(t, Y):
    pass
    
# TODO 3: 设置这俩事件的属性
# 警告：到底哪个事件需要 terminal = True 切断计算流？哪个仅仅只是记录？


# 启动链式求解引擎 (接力赛机制循环启动 10 次)
for _ in range(bounce_times):
    # TODO 4: 执行单次飞行积分直至命中落地事件
    # sol = solve_ivp(..., 传入当期起点与宽泛的终点, 传入当期状态, ...)
    pass
    
    # TODO 5: 将跑下来的坐标通过 .extend 追加到用来绘图的 all_t 和 all_y 大列表中
    
    # TODO 6: 收集并追加刚才由最高点事件触发并记下的坐标到追踪列表
    
    # TODO 7: 为下一次抛射重置并提供新初始条件！
    # 1. 拨表：新的一局起步时间必须为上一次砸地终止的确切瞬间
    # 2. 动能损耗公式：利用刚刚坠地接触时的瞬时速度，算出反弹的新出膛速度大小（原速度 0.9 倍并反向）
    
    
# TODO 8: 编写 matplotlib 逻辑绘制整根跳跃衰减长线，其中在各个最高点用红色特别点缀
