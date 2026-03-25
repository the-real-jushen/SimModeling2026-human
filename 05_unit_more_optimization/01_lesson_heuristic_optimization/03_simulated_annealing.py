# %% [markdown]
"""
# 模拟退火算法 (Simulated Annealing, SA)

## 1. 基本思想

模拟退火算法的思想源于固体的退火过程：将固体加热至足够高的温度，再缓慢冷却；升温时固体内部粒子随温度升高变为无序状，内能增加，而缓慢冷却又使得粒子逐渐趋于有序。冷却到低温时将达到这一低温下的内能最小状态。

![Simulated Annealing jumping out of local minima](https://upload.wikimedia.org/wikipedia/commons/d/d5/Hill_Climbing_with_Simulated_Annealing.gif)
*(图示：模拟退火算法在寻找最低点时，偶尔会允许“向上走”，从而跳出局部最优的坑，最终找到全局最优。)*

"""

# %% [markdown]
"""
## 3. 模拟退火算法 (Simulated Annealing, SA)

### 3.1 物理背景与算法思想

在冶金学中，**退火**是将金属加热到很高的温度，然后再缓慢冷却的过程。
*   **高温时**：金属内部的原子热运动剧烈，处于无序状态（能量高）。
*   **缓慢冷却时**：原子逐渐趋于有序，最终在常温下达到内能最小的稳定结晶状态。

**模拟退火算法**借用了这个思想来解决优化问题：
1.  **初始化**：设定一个很高的初始“温度” $T$，并随机生成一个初始解。
2.  **产生新解**：在当前解的附近随机产生一个微小的扰动，得到一个新解。
3.  **判断是否接受新解 (Metropolis 准则)**：
    *   如果新解比当前解**更好**（能量更低），则**100% 接受**新解。
    *   如果新解比当前解**更差**（能量更高），则**以一定的概率 $P$ 接受**新解。
        $$ P = \exp\left(-\frac{\Delta E}{T}\right) $$
        其中 $\Delta E$ 是新解与旧解的能量差（目标函数值之差）。
4.  **降温**：缓慢降低温度 $T$。
5.  **迭代**：在每个温度下重复步骤 2 和 3 多次，直到温度降到足够低，算法停止。

### 3.2 为什么能跳出局部最优？

关键在于那个**接受较差解的概率 $P$**：
*   **高温时（算法初期）**：$T$ 很大，$P$ 接近 1。算法非常“宽容”，即使新解很差也大概率接受。这使得算法具有很强的**探索 (Exploration)** 能力，能在整个解空间中大范围跳跃。
*   **低温时（算法后期）**：$T$ 很小，$P$ 接近 0。算法变得“挑剔”，几乎只接受更好的解。这使得算法转为**利用 (Exploitation)**，在局部进行精细搜索。

通过控制降温的速度，模拟退火算法巧妙地平衡了探索与利用。

"""

# %% [markdown]
"""
## 4. 代码实战：用模拟退火寻找全局最低点

我们将寻找函数 $f(x) = x^2 + 10\sin(5x) + 7\cos(4x)$ 在区间 $[-10, 10]$ 上的全局最小值。

"""

# %%
import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 1. 定义目标函数 (我们要寻找它的最小值)
def objective_function(x):
    return x**2 + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)

# 绘制目标函数的图像，看看它有多复杂
x_vals = np.linspace(-10, 10, 1000)
y_vals = objective_function(x_vals)

plt.figure(figsize=(10, 4))
plt.plot(x_vals, y_vals, label='Objective Function')
plt.title("Complex Function with Multiple Local Minima")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# 从图中可以看到，这个函数有很多个“坑”（局部最小值）。
# 如果用普通的梯度下降，很容易掉进某个坑里出不来。




# %%
# 2. 模拟退火算法实现

def simulated_annealing(objective, bounds, n_iterations, step_size, temp_init, cooling_rate):
    """
    模拟退火算法求解一维连续优化问题
    """
    # 随机生成初始解
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_eval = objective(best_x)
    
    # 当前解 (初始时等于最优解)
    curr_x, curr_eval = best_x, best_eval
    
    # 记录搜索过程，用于可视化
    history_x = [curr_x]
    history_eval = [curr_eval]
    
    # 当前温度
    curr_temp = temp_init
    
    for i in range(n_iterations):
        # 1. 产生新解：在当前解附近随机扰动
        candidate_x = curr_x + np.random.randn() * step_size
        
        # 确保新解在边界内
        candidate_x = np.clip(candidate_x, bounds[0], bounds[1])
        
        # 计算新解的目标函数值
        candidate_eval = objective(candidate_x)
        
        # 2. 判断是否接受新解
        # 如果新解更好，直接接受
        if candidate_eval < best_eval:
            best_x, best_eval = candidate_x, candidate_eval
            
        # 计算能量差 (目标函数值之差)
        diff = candidate_eval - curr_eval
        
        # Metropolis 准则：计算接受较差解的概率
        # 注意：如果 diff < 0 (新解更好)，exp(-diff/T) > 1，必然接受
        metropolis_prob = np.exp(-diff / curr_temp)
        
        # 生成一个 0-1 之间的随机数，如果小于概率 P，则接受新解
        if diff < 0 or np.random.rand() < metropolis_prob:
            curr_x, curr_eval = candidate_x, candidate_eval
            
        # 记录历史
        history_x.append(curr_x)
        history_eval.append(curr_eval)
        
        # 3. 降温 (指数降温)
        curr_temp = curr_temp * cooling_rate
        
    return best_x, best_eval, history_x, history_eval

# 3. 设置参数并运行算法
bounds = [-10, 10]
n_iterations = 1000   # 迭代次数
step_size = 1.0       # 扰动步长
temp_init = 100.0     # 初始高温
cooling_rate = 0.99   # 降温速率 (越接近1降温越慢)

# 运行模拟退火
best_x, best_eval, history_x, history_eval = simulated_annealing(
    objective_function, bounds, n_iterations, step_size, temp_init, cooling_rate
)

print(f"找到的全局最优解 x: {best_x:.4f}")
print(f"对应的最小函数值 f(x): {best_eval:.4f}")

# 4. 可视化搜索过程
plt.figure(figsize=(12, 6))

# 画出目标函数背景
plt.plot(x_vals, y_vals, label='Objective Function', alpha=0.5)

# 画出搜索轨迹 (用颜色深浅表示时间先后，越红越靠后)
colors = plt.cm.Reds(np.linspace(0.2, 1, len(history_x)))
plt.scatter(history_x, history_eval, c=colors, marker='.', s=20, label='Search Path')

# 标出最终找到的最优点
plt.plot(best_x, best_eval, 'g*', markersize=20, label='Found Global Minimum')

plt.title("Simulated Annealing Search Process")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

# 观察图表：红色的点代表算法的搜索轨迹。你可以看到，算法在初期（浅红色）在各个坑之间跳跃，
# 到了后期（深红色），它逐渐稳定在最低的那个坑里。




# %% [markdown]
"""
## 6. 进阶实战：模拟退火巧解“数独游戏”

模拟退火最强大的能力，不限于上面简单的连续维度，而在它的**离散组合排演能力**。这里我们利用鼎鼎大名的解数独任务展示如何构造。

**经典数独规则**：数字 1-9 在每一行、每一列、每一个用粗线分隔的 3x3 小宫格内只能出现一次。

### 算法设计三大核心步骤

1.  **初始解的巧妙设局**
    我们将 81 格切出 9 个 $3\times 3$ 小宫格（Block）。当我们填充初始未知的空白时，我们**针对每一个单独块内部，确保将还缺失的那几个数字完全独立、没有重复地填进去**。这样一开局就满足了 $3\times3$ 小内部无重复的约束。

2.  **邻域演化转移（微调）**
    如果在不同块里瞎交换数字就破坏了第一步。我们的每次扰动（Move）要求：随机挑一块 $3\times3$，交换内部我们填入的两个数字。
    这保证了整个寻优过程中，$3\times3$ 内部无重复这一规则永远不会被打破！

3.  **内能评判规则设计（寻找剩余冲突）**
    能量函数如何计算？除了 $3\times3$ 的约束满足了，只剩行和列。
    定义：**能量函数 (Energy) = -(所有行内不重复元素个数 + 所有列内不重复元素个数)**
    如果一完美行不重复元素是 9，完美的全部棋局分数则为 $-(9\times 9行 + 9\times 9列) = -162$。系统只要向 -162 收敛，数独就被解开了。

"""

# %%
import numpy as np
import random

# 初始化数独题目。_ = 0 代表需要补填的空位。
_ = 0
PROBLEM = np.array([
    1, _, _,  _, _, 6,  3, _, 8,
    _, _, 2,  3, _, _,  _, 9, _,
    _, _, _,  _, _, _,  7, 1, 6,

    7, _, 8,  9, 4, _,  _, _, 2,
    _, _, 4,  _, _, _,  9, _, _,
    9, _, _,  _, 2, 5,  1, _, 4,

    6, 2, 9,  _, _, _,  _, _, _,
    _, 4, _,  _, _, 7,  6, _, _,
    5, _, 7,  6, _, _,  _, _, 3,
])

def print_sudoku(state):
    border = "+-------+-------+-------+"
    print(border)
    rows = [state[i:i+9] for i in range(0, 81, 9)]
    for i, row in enumerate(rows):
        three = [row[j:j+3] for j in range(0, 9, 3)]
        print("| "+" | ".join(" ".join(str(x or "_") for x in one) for one in three) + " |")
        if (i + 1) % 3 == 0: print(border)

def coord(row, col): return row * 9 + col

def block_indices(block_num):
    firstrow = (block_num // 3) * 3
    firstcol = (block_num % 3) * 3
    return [coord(firstrow + i, firstcol + j) for i in range(3) for j in range(3)]

# --- 环节1 初态 ---
def initial_solution(problem):
    solution = problem.copy()
    for block in range(9):
        indices = block_indices(block)
        block_val = problem[indices]
        zeros = [i for i in indices if problem[i] == 0]
        to_fill = [i for i in range(1, 10) if i not in block_val]
        random.shuffle(to_fill)
        for index, value in zip(zeros, to_fill):
            solution[index] = value
    return solution

# --- 环节2 扰动转移 ---
def random_move(solution, problem):
    random_solution = solution.copy()
    block = random.randrange(9)
    indices = [i for i in block_indices(block) if problem[i] == 0]
    
    if len(indices) >= 2:
        m, n = random.sample(indices, 2)
        random_solution[m], random_solution[n] = random_solution[n], random_solution[m]
    return random_solution

# --- 环节3 计算系统的能量评估分数 ---
def calc_energy(solution):
    # 每列共有几个不同的数字
    def column_score(n): return -len(set(solution[coord(i, n)] for i in range(9)))
    # 每行共有几个不同的数字
    def row_score(n):    return -len(set(solution[coord(n, i)] for i in range(9)))
    # 总和，每一行每一列不同数字求和（取了负号以便于求最小）
    return sum(column_score(n) + row_score(n) for n in range(9))

# --- 环节4 接受准则 ---
# 计算接受较差解的概率
def probability(delta, T):
    return np.exp(-delta / T)

# 检查是否接收新的解
def deal(curr_solution, new_solution, delta, T):
    # Delta < 0说明新解能量更低（更好），直接接受
    if delta < 0:
        return new_solution, True
    # 否则，按照 Metropolis 准则依概率接受
    p = probability(delta, T)
    if p > random.random():
        return new_solution, True
    return curr_solution, False

# 打印当前探索状态
def print_status(trial, accept, best):
    print(f'尝试次数: {trial}, 接受新解次数: {accept}, 接受率: {(accept / trial):.2f}, 当前最优评估: {best}')

Tmax = 1.0          # 初始温度
Tmin = 0.05         # 终止温度
alpha = 0.95        # 每轮降温率
Lf = 4000           # 每个温度下的迭代（Markov链长度）

T = Tmax
solution = initial_solution(PROBLEM)
best_energy = calc_energy(solution)
best_solution = solution.copy()

print("\n--- 开始计算，初始盘面 ---")
print_sudoku(solution)
print(f"初始能量评估（最好应全消到达-162）: {best_energy}\n退火开始...")

loop_count = 0
trial, accept = 0, 0

while T >= Tmin:
    for i in range(Lf):
        energy = calc_energy(solution)
        # 更新当前的最优历史记录
        if energy < best_energy:
            best_energy = energy
            best_solution = solution.copy()
            if best_energy == -162: 
                break
                
        # 1. 产生随机扰动新解
        random_sol = random_move(solution, PROBLEM)
        # 2. 评估新解能量
        random_energy = calc_energy(random_sol)
        delta = random_energy - energy
        
        # 3. 决定是否接受新解
        solution, accepted = deal(solution, random_sol, delta, T)
        
        if accepted:
            accept += 1
        trial += 1
        
        # 每尝试10000次打印一次进度
        if trial % 10000 == 0:
            print_status(trial, accept, best_energy)
        
    if best_energy == -162: break
    T *= alpha
    loop_count += 1

print(f"\n-----------运算结束----------")
print(f"最终寻找最低能量: {best_energy}, 总尝试次数: {trial}")
if best_energy == -162:
    print("完美解开数独！")
else:
    print("未能完全避开冲突，得到近似最优解：")
print_sudoku(best_solution)




# %% [markdown]
"""
## 7. 使用现成的库 (simanneal) 解决数独

为了复用写好的退火逻辑，我们可以继承 `Annealer` 类并重写 `move()` 和 `energy()` 两个关键方法即可。这与我们上面纯手写的逻辑完全一致。

"""

# %%
# 这里如果报错没有 simanneal 库，可以在终端运行 `pip install simanneal` 安装
try:
    from simanneal import Annealer
    
    # 继承 Annealer 框架
    class Sudoku_Sq(Annealer):
        def __init__(self, problem):
            self.problem = problem
            # 调用我们之前的初态生成器
            state = initial_solution(problem)
            super().__init__(state)

        def move(self):
            """微调转移：在 3x3 宫格内随机交换两个元素"""
            self.state = random_move(self.state, self.problem)

        def energy(self):
            """能量评判：计算剩余的行和列的冲突"""
            score = calc_energy(self.state)
            if score == -162:
                # 找到解，触发早期退出机制
                self.user_exit = True  
            return score

    print("\n--- 使用 simanneal 库重新开始计算 ---")
    sudoku = Sudoku_Sq(PROBLEM)
    # copy_strategy = "method" 避免浅拷贝导致的混乱
    sudoku.copy_strategy = "method"
    
    # 设置退火参数
    sudoku.Tmax = 0.5
    sudoku.Tmin = 0.05
    sudoku.steps = 100000
    
    # 开始退火
    state, e = sudoku.anneal()
    
    print("\n库运算框架完成预测")
    print(f"最终能量 (目标是 -162): {e}")
    if e == -162:
        print("完美解开数独！")
    else:
        print("未能完全避开冲突，得到近似最优解：")
    print_sudoku(state)
    
except ImportError:
    print("\n[注意] 检测到你的环境中未安装 simanneal。可以跳过，或使用 pip install simanneal 安装后体验。")





# %% [markdown]
"""
### 知识测验

若你要提升求解速度，非常暴戾且强行地将模拟退火初始化系统的起始温度（例如 $T_0$）设定为极低（例如 0.001）。请问这一改动最有可能引起什么结果？


**选项:**
- A. 在早期也会使得模型几乎总是拒绝表现不太好的变异状态，变成纯贪心法，容易掉进局部陷阱且跑不出来
- B. 大大拉长了运算等待的探索时段周期，引发计算机无法负荷
- C. 会使得 Metropolis 判断式一直在高温接受差解导致算法失控
- D. 导致退火算法跳过迭代运算直接输出最优解

**正确答案:** A. 在早期也会使得模型几乎总是拒绝表现不太好的变异状态，变成纯贪心法，容易掉进局部陷阱且跑不出来

"""
