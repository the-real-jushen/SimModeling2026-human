# %% [markdown]
"""
# 遗传算法 (Genetic Algorithms, GA)

## 1. 基本思想与自然界类比

在前一节中，我们提到了**探索与利用 (Exploration and Exploitation)** 的平衡。为了实现这种平衡，科学家们从大自然的生物进化中汲取了绝佳的灵感。

达尔文的进化论告诉我们：“物竞天择，适者生存”。在自然界中，适应环境的优良基因通过繁衍被保留下来，不同个体的基因通过**交配（交叉）**产生新的组合，而基因在遗传过程中又会发生小概率的**突变（变异）**，从而产生更优秀的后代。
**遗传算法 (GA)** 正是基于这一思想，模拟生物进化过程的全局自适应概率搜索算法。

<div align="center">
<img src="images/pic30.png" width="80%">
<p>图1. 遗传算法的进化迭代循环</p>
</div>

### 核心概念映射
利用遗传算法求解问题时，需要将问题空间的解映射为生物种群：
*   **个体 / 染色体 (Chromosome)**：代表问题的一个**可行解**。
*   **基因 (Gene)**：代表解的一个特征或决策变量，多个基因组成一条染色体。编码方式通常有二进制编码（由0和1组成的串）和浮点数编码。
*   **种群 (Population)**：由多个个体（可行解）组成的集合。
*   **适应度 (Fitness)**：衡量个体优劣的标准。适应度函数由**目标函数**转化而来，适应度越高的个体代表目标函数值越优，越容易在“自然选择”中存活并繁衍。

"""

# %% [markdown]
"""
## 2. 经典例题：膜当劳积分兑换 (0-1背包问题)

为了让你直观感受到优化算法的作用，我们来看一个非常有生活气息的经典问题——**0-1背包问题**。

**问题描述：**
假设你有 **900** 的膜当劳积分（背包容量 $V=900$）。你可以用积分兑换下面 8 种商品。每种商品只能兑换 1 件（拿或不拿，即 0 或 1）。请问该如何挑选商品，使得在不超过 900 积分的前提下，兑换的商品**总售价最高**？

| 编号 | 积分要求(重量 $w$) | 售价(价值 $c$) |
| ---- | ---- | ---- |
| 0 | 99 | 9.5 |
| 1 | 139 | 9 |
| 2 | 300 | 19 |
| 3 | 280 | 15 |
| 4 | 188 | 10 |
| 5 | 210 | 11 |
| 6 | 240 | 12.5 |
| 7 | 260 | 13.5 |

### 暴力遍历法 (Brute Force)
由于目前只有 8 种商品，解的空间是 $2^8 = 256$ 种组合。我们完全可以把这 256 种组合全部试一遍，看看哪种总售价最高。你可以直接运行下方代码体验暴力破解：

"""

# %%
import time

c = 900  # 背包容量 (最大积分)
n = 8    # 物品数量
weight = [99, 139, 300, 280, 188, 210, 240, 260] # 消耗积分
value =  [9.5, 9, 19, 15, 10, 11, 12.5, 13.5]    # 价值

print("--- 开始暴力遍历 ---")
best_value = 0
best_cost = 0
best_plan_binary = 0

start_time = time.time()
# 0-1背包，2^n 种可能，使用整数的二进制位代表拿或不拿（即穷举：00000000 ~ 11111111）
for plan in range(2**n):
    current_value = 0
    current_cost = 0
    
    # 检查该 plan 的每一位，如果是 1 则代表取对应物品
    for i in range(n):
        # 掩码测试第 i 位是否为 1
        if plan & (1 << (n - i - 1)):
            current_cost += weight[i]
            current_value += value[i]
            
    # 只记录没超过 900 积分约束，且价值更高的方案
    if current_cost <= c and current_value > best_value:
        best_value = current_value
        best_cost = current_cost
        best_plan_binary = plan

end_time = time.time()

print(f"耗时: {end_time - start_time:.6f} 秒")
print(f"最优购买方案 (二进制): {bin(best_plan_binary)}")
print(f"最大价值: {best_value} , 消耗积分: {best_cost}")




# %% [markdown]
"""
### 维度灾难的体现
上面运行的代码瞬间就算出了最优解 `0b11100110` (代表拿商品 0, 1, 2, 5, 6)。
但请设想：如果备选商品增加到 **50种** 呢？
搜索空间变成了 $2^{50}$ 种组合，大约等于 $1.12 \times 10^{15}$ 次循环计算。原本瞬间完成的暴力遍历，可能需要一台普通电脑运行**几十年**。这就是我们在第一节提到的**维度灾难**。

当问题变得无法穷尽时，我们就需要祭出**遗传算法(GA)**了。

### 遗传算法核心概念深入解析

为了更好地理解遗传算法是如何工作的，我们需要深入探讨其背后的几个关键设计组件。

#### 1. 染色体编码 (Encoding)
利用遗传算法求解问题时，必须在目标问题的实际表示与染色体位串结构之间建立联系。这种映射过程称为**编码**，反向映射称为**解码**。
常用的编码方式主要有两种：
*   **二进制编码**：最常用的方法，将参数用 {0,1} 构成的位串表示（如本例 0-1 背包问题）。优点是简单易行，便于交叉和变异；缺点是连续函数离散化时存在映射误差，且对于高精度要求会导致串长度急剧增加。
*   **浮点数（实数）编码**：个体的每个基因采用给定范围内的某个浮点数表示。优点是不存在精度问题和映射误差，适合多维连续优化问题，但交叉变异操作需特殊设计保证不越界。

#### 2. 适应度函数 (Fitness Function)
适应度函数是衡量个体优劣的标准，**适应度值越大代表个体越好**。在自然选择的轮盘上，适应度决定了被抽中“繁衍后代”的概率。
*   适应度函数通常由目标函数转换而来。
*   由于基于概率进行选择，**适应度函数值一般不能小于零**。如果是求最小值问题，通常需要通过取倒数或用一个大数减去目标函数值来进行转换。

#### 3. 约束条件的处理
在实际问题（如背包的容量限制）中，常常遇到约束。遗传算法中处理约束一般有三种方法：
1.  **罚函数法 (Penalty Function)**：对于不满足约束的个体（无效解），人为计算时除以一个惩罚函数来**降低其适应度**，从而降低它被遗传到下一代的概率。
2.  **搜索空间限定法**：直接在生成和变异阶段强行限制，保证产生的任何个体都对应一个可行解。
3.  **可行解变换法**：在基因型到表现型的解码过程中，增加一套修复机制，将无效解强行“修复”或映射为满足约束的可行解。

#### 4. 遗传算子详细解析
遗传算法包含三个模拟生物进化核心机制的算子：选择、交叉和变异。

*   **选择操作 (Selection)**
    选择操作用来确定如何从当前群体中选取个体遗传到下一代，核心原则是**适应度较高的个体存活和繁衍的概率更大**。常用的方法有：
    *   **轮盘赌法 (Roulette Wheel Selection)**：个体被选中的概率与其适应度在总适应度中的比例成正比。就像一个轮盘，适应度越高的个体占据的扇区面积越大，指针停在其上的概率也就越大。公式为 $p_k = \frac{F(v_k)}{\sum F(v_j)}$。
    *   **排序选择法 (Rank Selection)**：先按适应度对所有个体排序，然后按排名人为分配既定的概率，再进行轮盘赌。这种方法能有效避免早期个别超优个体迅速占领整个种群而导致的“早熟”。
    *   **两两竞争法 / 锦标赛选择法 (Tournament Selection)**：随机在种群中挑选 $k$ 个个体进行“锦标赛”比较，适应度最高的胜出并进入下一代，重复此过程填满种群。该方法计算效率高且易于实现。

*   **交叉操作 (Crossover)**
    交叉是算法**利用已有信息 (Exploitation)** 的主要体现。通过让两个优秀个体的基因进行一定形式的混合，期望子代能继承双亲的优良特征。
    *   主要方式包括**单点交叉**（在染色体中间随机切一刀，双方交换后半段）、**多点交叉**（切多刀交换）或**均匀交叉**（按固定概率在每一个基因位上产生交换）。

*   **变异操作 (Mutation)**
    变异是算法**探索未知区域 (Exploration)**、防止陷入局部最优的关键。它以很小的概率将染色体编码串中的某些基因随机突变。
    *   在二进制编码中，变异常表现为**基本位变异**（某个位 0变1，或 1变0）。合理设定变异率非常重要：没有变异算法容易卡死在局部极小值；变异率过高则会退化成纯纯的随机搜索。

#### 5. 搜索终止条件
遗传算法是一种启发式迭代算法，通常我们需要设定一些条件来指引它何时结束搜索。只要满足以下任一条件，演化过程就会停止：
1.  **达到最大进化代数 $t$**：最普遍设定的终止条件，用来控制最大的运算时间，防止死循环。
2.  **适应度差异极小**：连续多次迭代（多代之间），最优个体的适应度相差在一个任意小的正数 $\epsilon$ 范围内，即 $0 < | F_{new} - F_{old} | < \epsilon$。这代表演化已经收敛，基本无法再找到更好的解。

#### 6. 关键参数控制
*   **种群数量**：太小不能提供足够的采样点，太大则收敛缓慢。
*   **交叉概率**：控制重组交换的频率。太大会破坏优良基因组合，太小则搜索停滞。
*   **变异概率**：增加多样性的关键。太小容易早熟，太大则退化为盲目的随机搜索。

有了这些核心理论武装，接下来我们将前面问题中的 `0b11100110` 二进制串视为**染色体 (Chromosome)**，用代码实现一个标准的遗传算法来解答这个问题。

"""

# %%
import numpy as np
import random
import time

# --- 遗传算法基本参数 ---
chromosome_size = 8      # 染色体长度（代表8个物品的拿取状态）
chromo_pop_size = 100    # 种群规模（每一代有100个个体在进化）
selection_num = 60       # 选择生存下来的个体数
mutation_num = 30        # 每一代发生变异的个体数量
max_iter = 200           # 最大迭代进化代数

# 1. 初始化种群 (Initialization)
def init_population(pop_size, chromosome_size):
    chromosome_states = []
    for _ in range(pop_size):
        # 随机产生二进制个体，范围 0 到 255
        chromosome = random.randint(0, (1 << chromosome_size) - 1)
        chromosome_states.append(chromosome)
    return chromosome_states

# 2. 计算适应度 (Fitness Evaluation)
def calc_fitness(chromosome_states):
    fitness_list = []
    for chromosome in chromosome_states:
        value_sum, weight_sum = 0, 0
        for i in range(chromosome_size):
            mask = 1 << (chromosome_size - i - 1)
            # 如果某位为1，说明拿了该物品
            if chromosome & mask:
                weight_sum += weight[i]
                value_sum += value[i]
        fitness_list.append([value_sum, weight_sum])
    return fitness_list

# 3. 环境约束淘汰 (处理越界惩罚：超过积分上限直接抹杀)
def filter_population(chromosome_states, fitness_list):
    acceptable_fitness = []
    acceptable_chromosome = []
    for i, f in enumerate(fitness_list):
        if f[1] <= c: # 重量(积分)没有超载
            acceptable_fitness.append(f)
            acceptable_chromosome.append(chromosome_states[i])
    return acceptable_fitness, acceptable_chromosome

# 4. 锦标赛选择法 (Tournament Selection)
def select(fitness_list, chromosome_list):
    survivor_fitness = []
    survivor_chromosome = []
    for _ in range(selection_num):
        num = len(fitness_list)
        idx1 = random.randint(0, num - 1)
        idx2 = random.randint(0, num - 1)
        # 两人竞争，价值更高的活下来
        if fitness_list[idx1][0] > fitness_list[idx2][0]:
            survivor_fitness.append(fitness_list[idx1])
            survivor_chromosome.append(chromosome_list[idx1])
        else:
            survivor_fitness.append(fitness_list[idx2])
            survivor_chromosome.append(chromosome_list[idx2])
    return survivor_fitness, survivor_chromosome

# 5. 交叉繁衍 (Crossover)
def crossover(chromosome_states, added_num):
    next_gen_chromosomes = []
    for _ in range(added_num):
        parent1 = random.choice(chromosome_states)
        parent2 = random.choice(chromosome_states)
        
        # 随机选取交叉的切分位置
        pos = random.choice(range(chromosome_size))
        # 组装掩码：将父亲的一半与母亲的一半融合
        mask2 = (1 << (chromosome_size - pos)) - 1
        mask1 = (1 << chromosome_size) - 1 - mask2
        
        child = (parent1 & mask1) + (parent2 & mask2)
        next_gen_chromosomes.append(child)
    return next_gen_chromosomes

# 6. 变异 (Mutation)
def mutate(chromosome_states):
    for _ in range(mutation_num):
        idx = random.randint(0, len(chromosome_states)-1)
        pos = random.choice(range(chromosome_size))
        # 用异或(^)操作使该位翻转 (0变1，1变0)
        mask = 1 << pos
        chromosome_states[idx] = chromosome_states[idx] ^ mask
    return chromosome_states

# --- 执行完整的遗传演化过程 ---
print("\n--- 开始遗传算法 ---")
start_time = time.time()
population = init_population(chromo_pop_size, chromosome_size)

for current_iter in range(max_iter):
    # a. 算适应度
    fitnesses = calc_fitness(population)
    # b. 过滤掉超重的（天敌淘汰）
    fitnesses, population = filter_population(population, fitnesses)
    # c. 适者生存选择
    if len(fitnesses) > selection_num:
        fitnesses, population = select(fitnesses, population)
    # d. 繁衍后代补齐数量
    population += crossover(population, chromo_pop_size - len(population))
    # e. 以一定概率变异
    population = mutate(population)

end_time = time.time()

# 筛选并找出最终的最优个体
final_fitnesses = calc_fitness(population)
final_fitnesses, final_population = filter_population(population, final_fitnesses)

best_fit = max(final_fitnesses, key=lambda x: x[0])
best_idx = final_fitnesses.index(best_fit)

print(f"演化耗时: {end_time - start_time:.6f} 秒")
print(f"繁衍的最优方案 (二进制): {bin(final_population[best_idx])}")
print(f"最大价值: {best_fit[0]} , 消耗积分: {best_fit[1]}")

# 补充观察：你可能会发现，对于这么小的 8维问题，遗传算法的花费时间大于单纯的暴力遍历。
# 但随着问题的急剧变大（例如物品变成100个），暴力遍历即使运行一万年也出不来结果，
# 而遗传算法经过数十代的进化，几秒内依然能给出高度近似最优的组合！





# %% [markdown]
"""
### 知识测验 1

在刚刚的遗传算法实现中，保证算法不完全陷入父辈提供的特征循环（跳出局部最优极小值陷阱），代表了优化思想中“Exploration (探索)” 行为的，是三大算子中的哪一个？


**选项:**
- A. 选择 (Selection)
- B. 交叉 (Crossover)
- C. 变异 (Mutation)
- D. 惩罚 (Penalty)

**正确答案:** C. 变异 (Mutation)

"""

# %% [markdown]
"""
### 知识测验 2

在遗传算法的选择操作 (Selection) 中，哪种方法的核心思想是“根据个体在环境中的相对适应度比例来分配其繁衍后代的概率区间”，类似于转动一块被切分成不同面积扇区的圆盘进行抽奖？


**选项:**
- A. 锦标赛选择法 (Tournament Selection)
- B. 轮盘赌法 (Roulette Wheel Selection)
- C. 排序选择法 (Rank Selection)
- D. 交叉重组法 (Crossover)

**正确答案:** B. 轮盘赌法 (Roulette Wheel Selection)

"""
