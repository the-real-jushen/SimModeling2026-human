# %% [markdown]
"""
前两节我们学习了模拟退火和遗传算法。这节我们将探讨另一种大自然启发的算法——蚁群优化算法 (Ant Colony Optimization, ACO)。为了更好体现它的优势，我们将引入非常经典且困难的难题：旅行商问题 (TSP)。

# 蚁群系统与旅行商问题 (TSP)

## 1. 旅行商问题 (Travelling Salesman Problem, TSP)

### 1.1 简介
旅行商问题是一个十分经典的组合优化问题：一个售货员必须访问 $n$ 个城市，恰好访问每个城市一次，并最终回到出发城市。各个城市间有一条明确的路径长度（距离）。售货员希望使整个旅行费用（即总距离）最低。

与单纯寻找两点间最短路径的 Dijkstra 算法或 A* 算法不同，TSP 问题要求的是寻找一条**全局最短的闭合回路 (Hamilton 回路)**。

![](images/pic29.png)

### 1.2 数学模型
记赋权图 $G=(V,E)$，$V$为顶点集，$E$为边集，各顶点间的距离 $d_{ij}$ 已知。设决策变量：
$$
x_{ij}=
\begin{cases}
1, &若(i, j)在回路路径上 \\
0, &其他
\end{cases}
$$
则经典TSP问题可写为如下数学规划模型：
$$
\min\quad Z=\sum_{i=1}^{n} \sum_{j=1}^{n} d_{ij}x_{ij} \\
s.t. \begin{cases}
\sum_{j=1}^nx_{ij}=1, &i \in V \\
\sum_{i=1}^nx_{ij}=1, &j \in V \\
\sum_{i\in S}\sum_{j\in S}x_{ij} \le |S| - 1, &\forall S\subset V, 2\le |S| \le n-1 \\
x_{ij} \in \{0,1\}
\end{cases}
$$

模型中，n 为图中的顶点数，前两个约束表示对每个点仅有一条边进和一条边出，第三个约束保证没有任何子回路解的产生。满足前三个约束的解构成了一条 Hamilton 回路。
当 $d_{ij}=d_{ji} \quad (i,j\in V)$ 时称为**对称型TSP**，否则为**非对称型TSP**。

"""

# %% [markdown]
"""
TSP 有多难？我们先来试试用最直观的方法：穷举法（暴力搜索），把所有的走法都尝试一遍看看效果。

## 2. 穷举法尝试

遍历每一种路线排列，计算其总距离，然后寻找其中最短的距离。
假设有 $N$ 个城市，除了起点外，中间节点的排列方式总共有 $(N-1)!$ 种。算法的时间复杂度高达 $O(N!)$。当城市稍微多一些时，计算量会发生“组合爆炸”。

下面我们产生 10 个随机城市，看看暴力求解需要花多长时间。

"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import math
import time


# --- 1. 产生地图 ---
def generateMap(num_points):
    # 随机生成坐标
    points_coordinate = np.random.rand(num_points, 2)
    # 快速计算出任意两点间的距离矩阵
    distance_matrix = spatial.distance.cdist(
        points_coordinate, points_coordinate, metric='euclidean')
    return points_coordinate, distance_matrix

# --- 2. 穷举法生成下一条路径的排列组合算法 ---
def nextRoute(route):
    """
    寻找当前排列字典序的下一个排列
    """
    route_array = np.array(route)
    for i in range(len(route_array)):
        if np.all(route_array[:-(i+2):-1] == np.sort(route_array[:-(i+2):-1])):
            continue
        swap1 = route_array[-(i+1)]
        swap = np.sort(route_array[-i:])
        for j in range(len(swap)):
            if swap[j] > swap1:
                route_array[-(i+1)] = swap[j]
                swap[j] = swap1
                route_array[-i:] = np.sort(swap)
                return route_array.tolist()
    return route_array.tolist()

# --- 3. 路径评估函数 ---
def evaluate(route, distMat):
    start = 0
    dist = 0
    # 注意输入的 route 是从第1个点开始到最后一个点的排列，实际起点为0
    for nextCity in route:
        dist += distMat[start, nextCity]
        start = nextCity
    # 走完一圈还要再回到起点(0)
    dist += distMat[start, 0]
    return dist

# 设置测试点数
num_city = 10
city_coordinates, distance_matrix = generateMap(num_city)

print(f"城市数量: {num_city}, 总需遍历路线数: {math.factorial(num_city-1)}")

# 初始化路径
route = list(range(1, num_city))
bestRoute = []
dists = []
bestDist = np.inf
route2 = []

print("穷举法开始计算...")
startTime = time.time()

# 开始遍历所有的排列
while route2 != route:
    dist = evaluate(route, distance_matrix)
    dists.append(dist)
    if dist < bestDist:
        bestDist = dist
        bestRoute = route.copy()
        
    route2 = route.copy()
    route = nextRoute(route)

endTime = time.time()
print(f"穷举计算耗时: {endTime - startTime:.4f} 秒")

# 插入起点和终点，方便画图
bestRoute = [0] + bestRoute + [0]

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(city_coordinates[bestRoute, 0], city_coordinates[bestRoute, 1], 'o-r')
ax[0].set_title(f"Best Route (Dist: {bestDist:.4f})")
for idx, (x, y) in enumerate(city_coordinates):
    ax[0].text(x, y, f' {idx}', fontsize=12)
    
ax[1].plot(dists, alpha=0.5)
ax[1].set_title("Exhaustive Search Process")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Distance")
plt.show()

print('穷举法得到访问顺序：', ' -> '.join([str(p) for p in bestRoute]))
print(f'最短距离：{bestDist:.4f}')




# %% [markdown]
"""
可以看出，仅仅只有 10 个城市，计算量就非常庞大了。为了快速获得高质量的解，我们引入蚁群算法。

## 3. 蚁群算法 (Ant Colony Optimization, ACO)

算法由 Marco Dorigo 于 1992 年提出，灵感来源于真实蚂蚁寻找食物的过程。
它具有**分布计算、信息正反馈**和**启发式搜索**的特征，是一种启发式全局优化算法。

### 3.1 基本思想
*   蚂蚁在寻找食物走过的路径上会释放**信息素**(Pheromone)，其他蚂蚁会察觉并受影响。
*   路径上走过的蚂蚁越多，留下的信息素越浓。
*   信息素越浓，越容易吸引后来的蚂蚁（产生正反馈）。但同时，信息素也会随着时间的推移而**挥发**（防止陷入死锁）。

这代表了一种**探索 (Exploration)**与**利用 (Exploitation)**的平衡：有更近的路 (启发的贪心)，还有信息素高的路 (经验积累，利用)，并且保留了一定的概率随机盲目走 (全局探索)。

"""

# %% [markdown]
"""
我们来看一下蚁群算法运作时的核心公式是怎么构建出来的。它分为选择概率与信息素更新两部分。

### 3.2 关键公式机制

由于是 TSP 问题，初始时刻蚂蚁被随机放到不同的城市，每条边上的信息素设为相同的值。

**① 路线选择概率（转移概率）**

时刻 $t$，蚂蚁 $k$ 从城市 $i$ 走向 未访问城市 $j$ 的概率为：
$$
p_{ij}^k(t) = \frac{\left[ \tau_{ij}(t) \right]^\alpha \cdot \left[ \eta_{ij}(t) \right]^\beta}{\sum_{s\in allow_k} \left[ \tau_{is}(t) \right]^\alpha \cdot \left[ \eta_{is}(t) \right]^\beta}
$$
*   $\tau_{ij}$: 路径上的**信息素浓度**。
*   $\eta_{ij}$: 启发函数（通常取距离的倒数 $\frac{1}{d_{ij}}$），距离越近，渴望程度越高。
*   $\alpha$: **信息素因子**。越大代表蚂蚁越倾向于走“前人的老路”。
*   $\beta$: **启发函数因子**。越大代表蚂蚁越短视（贪心找近路）。

![](images/pic32.png)

**② 信息素的挥发与更新**

经过一轮完整巡回后，路径上的信息素会蒸发一部分，同时获得走过这段路的蚂蚁散发的新信息素：
$$
\begin{cases}
\tau_{ij}(t+1) = (1-\rho) \cdot \tau_{ij}(t) + \Delta \tau_{ij}, \quad 0 < \rho < 1 \\
\Delta \tau_{ij} = \sum_{k=1}^m \Delta \tau_{ij}^k
\end{cases}
$$
*   $\rho \, (0<\rho<1)$: **信息素挥发因子**。
*   $\Delta \tau_{ij}$: 所有蚂蚁在此边释放的信息素增量之和。具体某只蚂蚁在该路径上释放量为：
$$
\Delta \tau_{ij}^k =
\begin{cases}
\frac{Q}{L_k}, & 若蚂蚁k从城市i访问城市j \\
0, & 否则
\end{cases}
$$
（$L_k$ 是它刚跑完的整圈总长度，$Q$是常数）。因此，如果蚂蚁没有经过这条路径，它的信息素释放量就是0；而总路程越短的成功蚂蚁，在它走过的路径上释放的信息素越浓厚！

### 3.3 再聊聊 Exploration 与 Exploitation 的哲学
对于 TSP 问题，蚁群的整体思路就是随机在某些节点上放置蚂蚁找下一个节点。这个选择也不完全是随机的，而是根据距离远近来随机（越近的被选择概率越大）。
如果此时只是不断重复直到所有的蚂蚁都完成了一个 loop，这显然不是最优的：
- 这里面有 **Exploration（探索）**：随机放置蚂蚁，以概率随机选择路线。
- 也有 **Exploitation（利用）**：近路会比较容易被选中。
如果蚂蚁仅仅贪婪地只选择最短的那一条小命脉，那就是纯贪心算法了，被卡在局部最优里出不来。

为了平衡这个 EE dilemma（探索与利用困境），我们加入了**“信息素”**机制。
蚂蚁跑完一圈以后，所有的路径被评估。走得好的蚂蚁在那条路线上的信息素会沉积得更多。下一次当别的蚂蚁再来到这个节点算概率时，信息素多且路程短的路径脱颖而出。

### 3.4 参数对优化的影响
*   **蚂蚁数量 $m$**：太大会导致所有路长满了信息素，陷入平均；太小会导致个别路径过早失去探索可能。
*   **信息素挥发因子 $\rho$**：挥发太慢会使得旧且差的信息素干扰目前寻找，容易陷入局部最优；太快则遗忘太快，难以收敛。
*   **信息素常数 $Q$**：越大则信息素收敛速度越快，但也容易陷入局部最优。

### 3.4 优缺点小结
*   **优点**：分布式计算能力强，启发式概率搜索不易陷入局部最优，具有正反馈机制。
*   **缺点**：初期时信息素一致，收敛较慢；参数如果不当（例如挥发因子过低），可能陷于“死锁”或局部极值。

"""

# %% [markdown]
"""
理论懂了，下面我们使用真实代码为那 10 个随机城市的网络执行一次多轮蚂蚁巡线派送。

## 4. 蚁群算法求解 TSP 代码实战

我们将利用刚才随机生成的城市坐标，启动蚁群寻找更优的路径。

"""

# %%
# ----------- 算法参数设置 -----------
num_ant = 6          # 蚂蚁数量
alpha = 2            # 信息素重要程度因子 (影响老经验)
beta = 5             # 启发函数重要程度因子 (影响抄近道)
rho = 0.1            # 信息素挥发速度
Q = 1                # 信息素常数
max_iter = 150       # 蚁群外出探测迭代总轮数

# --- 启发函数 (距离的倒数) ---
# 为了防止除以自己(距离为0)，我们在对角线加了一个大数 1e10
eta_table = 1.0 / (distance_matrix + np.diag([1e10] * num_city))

# --- 初始信息素矩阵 (都一样设为1) ---
tau_table = np.ones((num_city, num_city))

# 临时路径记录: (蚂蚁数 x 城市数)
path_table = np.zeros((num_ant, num_city)).astype(int)

# 数据用于可视化
length_best = np.zeros(max_iter)
path_best = np.zeros((max_iter, num_city))

print("蚁群算法开始执行...")
startTime = time.time()

for iter_idx in range(max_iter):
    # 1. 随机分布蚂蚁的初始城市起点
    if num_ant < num_city:
        starts = np.random.permutation(range(num_city))[:num_ant]
    else:
        starts = np.random.permutation(range(num_ant)) % num_city
        
    path_table[:, 0] = starts
    lengths = np.zeros(num_ant)

    # 2. 每只蚂蚁开始遍历城市
    for antNo in range(num_ant):
        visiting = path_table[antNo, 0]
        unvisited = set(range(num_city))
        unvisited.remove(visiting)

        # 走剩余的 num_city-1 步
        for j in range(1, num_city):
            unvisited_list = list(unvisited)
            prob_trans = np.zeros(len(unvisited_list))

            # 计算前往剩余城市各自的概率权重
            for k in range(len(unvisited_list)):
                target = unvisited_list[k]
                prob_trans[k] = (tau_table[visiting][target] ** alpha) * (eta_table[visiting][target] ** beta)

            # 轮盘赌算法选择真正要走的路
            cumsum_prob_trans = (prob_trans / prob_trans.sum()).cumsum()
            cumsum_prob_trans -= np.random.rand()
            # 找到首个累计概率大于0的城市索引
            chosen_target_idx = list(cumsum_prob_trans > 0).index(True)
            chosen_city = unvisited_list[chosen_target_idx]

            # 移动蚂蚁并更新数据
            path_table[antNo, j] = chosen_city
            unvisited.remove(chosen_city)
            lengths[antNo] += distance_matrix[visiting][chosen_city]
            visiting = chosen_city

        # 蚂蚁最后还要返回起点！
        lengths[antNo] += distance_matrix[visiting][path_table[antNo, 0]]

    # 3. 本轮结束，找出此轮表现最好的蚂蚁记录
    current_best_len = lengths.min()
    current_best_path = path_table[lengths.argmin()].copy()
    
    if iter_idx == 0 or current_best_len < length_best[iter_idx - 1]:
        length_best[iter_idx] = current_best_len
        path_best[iter_idx] = current_best_path
    else:
        # 如果这轮没出好成绩，最好的依旧是之前的
        length_best[iter_idx] = length_best[iter_idx - 1]
        path_best[iter_idx] = path_best[iter_idx - 1].copy()

    # 4. 全局更新信息素！
    change_pheromone_table = np.zeros((num_city, num_city))
    for antNo in range(num_ant):
        for j in range(num_city - 1):
            _from = path_table[antNo, j]
            _to = path_table[antNo, j + 1]
            change_pheromone_table[_from][_to] += Q / lengths[antNo]
            
        # 回原点的那一步也要加释放信息素
        _from = path_table[antNo, num_city - 1]
        _to = path_table[antNo, 0]
        change_pheromone_table[_from][_to] += Q / lengths[antNo]
        
    # 信息素 = 留存的(挥发后) + 新增的释放
    tau_table = (1 - rho) * tau_table + change_pheromone_table

endTime = time.time()
print(f"蚁群算法计算耗时: {endTime - startTime:.4f} 秒")

# 将最终的最优路径加上闭环尾巴
final_best_route = [int(p) for p in path_best[-1]] + [int(path_best[-1][0])]
final_best_coord = city_coordinates[final_best_route, :]

print('蚁群推荐访问顺序：', ' -> '.join([str(p) for p in final_best_route]))
print(f'最优距离发现：{length_best[-1]:.4f}')

# 5. 绘图对比
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(final_best_coord[:, 0], final_best_coord[:, 1], 'o-b', label='ACO route')
ax[0].plot(city_coordinates[bestRoute, 0], city_coordinates[bestRoute, 1], '--r', alpha=0.3, label='Exhaustive Best')
ax[0].set_title(f"ACO Route (Dist: {length_best[-1]:.4f})")
ax[0].legend()
for idx, (x, y) in enumerate(city_coordinates):
    ax[0].text(x, y, f' {idx}', fontsize=12)

ax[1].plot(length_best, lw=2, c='b')
ax[1].set_title("Convergence of Shortest Distance")
ax[1].set_xlabel('Iteration Count')
ax[1].set_ylabel('Shortest distance')
plt.show()




# %% [markdown]
"""
通过知识测验，检验下你对挥发因子的理解到底透不透彻！

### 知识测验

在蚁群算法机制中，关于“信息素挥发因子 $\rho$（或存留率 $1-\rho$）”的影响，下列选项中哪项表述是算法初期调参时最需要注意的？（请参悟如果在参数部分它挥发过慢容易造成什么结果。）


**选项:**
- A. 过高容易导致遗忘新探索区域，过低容易累积旧信息导致停滞不前陷入死锁
- B. 挥发速度对搜索速度和全局搜索能力没任何真实影响
- C. 过快的挥发反而会让它避免陷入由于次优解积累导致的局部极值
- D. 信息素挥发因子越大代表探索新区域的纯概率上升

**正确答案:** A. 过高容易导致遗忘新探索区域，过低容易累积旧信息导致停滞不前陷入死锁

"""

# %% [markdown]
"""
再来测验一下参数 $\alpha$ 和 $\beta$ 对探索与利用的影响。

### 进阶测验

对于蚁群算法中起到调节 Exploration（探索）与 Exploitation（利用）平衡的两个核心参数：信息素因子 $\alpha$ 和启发函数因子 $\beta$。下列说法中正确的是？


**选项:**
- A. 令 $\alpha = 0$ 会使算法变成纯贪心法（每次只挑最近的走），丧失利用信息素正反馈的特性。
- B. 令 $\beta = 0$ 会使算法只考虑启发距离，忽略信息素。
- C. $\alpha$ 和 $\beta$ 会同步增加蚁群在寻找最短路径时的随机盲目探索能力。
- D. 信息素因子的增加必定能极大加速全局最优的发现，绝不会陷入局部陷阱。

**正确答案:** A. 令 $\alpha = 0$ 会使算法变成纯贪心法（每次只挑最近的走），丧失利用信息素正反馈的特性。

"""
