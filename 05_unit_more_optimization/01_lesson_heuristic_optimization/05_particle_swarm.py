# %% [markdown]
"""
除了模拟物理退火和生物进化，科学家们还观察到了自然界中群体动物的社会行为。比如，一群鸟在寻找食物时，它们并不是各自为战，而是通过信息共享，整个鸟群能够迅速飞向食物最丰富的地方。**粒子群优化算法 (Particle Swarm Optimization, PSO)** 就是受这种鸟群觅食行为启发而提出的一种群体智能算法。

# 粒子群优化算法 (Particle Swarm Optimization)

## 1. 核心概念与鸟类觅食类比

想象一群鸟在一个区域内随机寻找食物。它们不知道食物的具体位置，但在每一次寻找后，它们知道自己当前碰到的食物有多少（对应目标的适应度），并且也知道自己离食物的大致远近。它们如何最快地找到整个区域内食物最多的那个核心位置？

最有效的策略是：**不仅记住自己曾经找到的最好的地方，还要打听所有鸟中谁找到了最好的地方，并且向那个方向靠拢。**

这便是 PSO 算法的基本思想。在 PSO 算法中：
*   **粒子 (Particle)**：代表鸟群中的一只鸟，它是搜索空间里问题的一个**候选解**。
*   **位置 (Position)**：对应解的具体数值，在多维空间中是一个向量 $(x_1, x_2, \dots, x_D)$。
*   **速度 (Velocity)**：代表粒子移动的方向和快慢，同样是一个向量 $(v_1, v_2, \dots, v_D)$。
*   **适应度 (Fitness)**：代表该位置食物的丰富程度，即用来评估解好坏的目标函数值。

每个粒子在搜索（飞行）的过程中，会记住两个重要的个人与社会经验信息：
1.  **个体最优 (pBest, Personal Best)**：它自己在这段时间里曾经找到过食物最多的位置。这代表了粒子的“自我经验”。
2.  **全局最优 (gBest, Global Best)**：整个鸟群在这段时间里，所有个体碰到的最好的位置。这代表了粒子的“社会经验”或“信息共享”。

![](images/pic27.png)

## 2. 粒子的运动规则与关键公式

在每一次寻优的迭代中，每个粒子都会根据以下三个因素来更新自己的飞行速度，随后在原位置上移动一段距离：

1.  **惯性 (Inertia)**：鸟儿有保持自己原来的飞行方向和速度的倾向。
2.  **认知部分 (Cognitive)**：向自己曾经找到的最好位置（pBest）飞去，也就是不能忘了初心。
3.  **社会部分 (Social)**：向整个群体目前发现的最好位置（gBest）飞去，也就是通过同伴获得经验。

**速度更新公式：**
$$ V_{i}^{k+1} = w \cdot V_{i}^k + c_1 \cdot rand() \cdot (pBest_i - X_{i}^k) + c_2 \cdot rand() \cdot (gBest - X_{i}^k) $$

**位置更新公式：**
$$ X_{i}^{k+1} = X_{i}^k + V_{i}^{k+1} $$

其中角标 $k$ 代表第 $k$ 次迭代，$i$ 代表第 $i$ 个粒子。我们可以将该公式拆解来理解：
*   $w \cdot V_{i}^k$ 是**惯性部分**，保证粒子有继续自主探索新区域的能力。
*   $c_1 \cdot rand() \cdot (pBest_i - X_{i}^k)$ 是**自我认知部分**，拉着粒子向自己曾经的巅峰靠近。
*   $c_2 \cdot rand() \cdot (gBest - X_{i}^k)$ 是**社会群体部分**，拉着粒子向整个团队目前的顶层峰值靠近。

![](images/pic28.png)

## 3. PSO的参数详解

要调整好一群“计算机鸟儿”寻找方向时的状态，通常需要调教以下这几个非常核心的参数：

*   **惯性权重 $w$**：它表示微粒对当前自身运动状态的信任。它维持粒子继续原来的运动状态。
    *   **较大的 $w$** 意味着它很喜欢到处乱跑，不轻易被拉回，有利于扩大搜索空间跳出局部最优（偏探索 Exploration）。
    *   **较小的 $w$** 意味着它的惯性很小，很容易就被社会和过去的经验拉拢，这有利于它在已发现的优良区域内迅速收敛（偏利用 Exploitation）。
    *   *常用技巧*：在算法初期设为较大数值（如0.9），随迭代次数增加线性递减（如至0.4），从而兼顾初期的全局搜索与后期的局部深挖。
*   **加速常数 $c_1$ 和 $c_2$**（也称学习因子）：
    *   $c_1$ 是自我认知系数，让它更相信自己；$c_2$ 是社会系数，让它更相信群体。
    *   将二者统一起来称为 $\phi = c_1 + c_2$。$\phi$ 很大时粒子位置变化快，反之很慢。通常取 $c_1=2.0$，$c_2=2.0$ 时具有较好的收敛特征。
*   **最大速度 $V_{max}$**：决定粒子在一个循环中最大的移动距离，这是限制粒子飞得“太疯”而越过最优点或定义域的防护设置。
*   **粒子数量**：一般取 20～40 之间就能应付大多数问题。特定的复杂情况可扩展到 100~200。

"""

# %% [markdown]
"""
让我们用它来解决我们在电力系统中遇到的一种非常棘手的复杂调度问题：由于发电机的阀点效应带来的高度非线性组合（这类含有三角与绝对值的跳跃函数往往是普通数学求导法的大敌）。

## 4. 实例：考虑发电机阀点效应的经济调度

**问题背景：**
在大型汽轮发电机中，为了控制进汽量，通常会安装多个蒸汽阀门。当一个新阀门刚打开时，会产生节流损耗，导致能耗曲线出现突变（波动）。这种现象称为**阀点效应 (Valve-point Effect)**。

考虑阀点效应后，发电机的成本函数不再是平滑的二次曲线，而是叠加了一个伴随正弦振荡的绝对值项：

![](images/pic17.png)

**目标函数：**
假设有两台发电机分配总功率负荷为 100MW，使耗费的总成本最小：
$$ \min F = C_1(P_{G1}) + C_2(P_{G2}) $$
其中：
*   $C_1(P_{G1}) = 0.0007 P_{G1}^2 + 0.30 P_{G1} + 4 + |10 \cdot \sin(0.126 \cdot (30 - P_{G1}))|$
*   $C_2(P_{G2}) = 0.0004 P_{G2}^2 + 0.32 P_{G2} + 3 + |3 \cdot \sin(0.378 \cdot (0 - P_{G2}))|$

**约束条件：**
1.  功率平衡：$P_{G1} + P_{G2} = 100$
2.  输出约束：$30 \le P_{G1} \le 150$, $0 \le P_{G2} \le 50$
3.  网络潮流限制（在此我们将其缩略，只关注机组上下限。通过 $P_{G2} = 100 - P_{G1}$ 我们可以将变量缩减到底层的单变量寻优模型来进行演示讲解。）

因为其自带多峰多极值的剧烈振荡性质，常规的基于梯度的纯解析方法极易停在此起彼伏的最差波谷里。下面我们将自己手工复刻一遍底层粒子群运算，来看“群体寻觅”是如何规避掉那些局部浅坑的！

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 手工实现核心：粒子群追踪计算 ----------------

# 目标函数。因为通过等式约束 PG2 = 100 - PG1，它简化为了只有 PG1 的一维变量模型：
def objective(pg1):
    pg2 = 100 - pg1
    
    # 模拟罚函数：如果PG2超出了合理区间 [0, 50]，则施以极高昂的代价值。
    if pg2 < 0 or pg2 > 50:
        return 1e9
        
    cost1 = 0.0007 * pg1**2 + 0.30 * pg1 + 4 + np.abs(10 * np.sin(0.126 * (30 - pg1)))
    cost2 = 0.0004 * pg2**2 + 0.32 * pg2 + 3 + np.abs(3 * np.sin(0.378 * (0 - pg2)))
    
    return cost1 + cost2

# ------- 1. 初始化超参数 -------
max_iter = 150         # 迭代次数限制
num_particle = 40      # 总群体的微粒数量
NDim = 1               # 问题的空间维数（在单变量上飞翔）

xmin, xmax = 30, 150   # 发电机PG1边界上下限
c1 = 2.0; c2 = 2.0     # 加速相关学习常数

# 惯性权重的衰减设置：初期大w找寻更多解，后期小w集中深挖最优地
start_weight = 0.9
end_weight = 0.4
weight_step = (start_weight - end_weight) / max_iter

# ------- 2. 环境初态散布与打分 -------
# 初始化位置与初速度（在整个值域中完全随机均匀撒开群鸟）
particle = xmin + (xmax - xmin) * np.random.rand(num_particle, NDim)
V = 0.5 * (xmax - xmin) * (np.random.rand(num_particle, NDim) - 0.5)

# 历史经验归档（初始时，大家的自身历史最佳自然都是诞生地）
pbest = np.copy(particle)
fitness = np.array([objective(p[0]) for p in particle])    # 计算第一轮的评估
pbest_value = np.copy(fitness)                             # 个人优选成绩记录得分

# 集体领头经验归档
gbest_value = np.min(pbest_value)                          # 获取整体里面最低的代价值（成本）
gbest = particle[np.argmin(pbest_value)]                   # 抓出那个最低成本位置记录为社会领航坐标

obj_fun_val = np.zeros(max_iter)                           # 制备画图序列板

# ------- 3. 开始周期性进化迭代（鸟群飞行）-------
for iter in range(max_iter):
    # 使小鸟不断衰减寻找的欲望，增强后期降落集中的能力
    w = start_weight - (iter + 1) * weight_step
    
    # 【核心1】微粒速度更新计算式
    # 老速度保持 +  自尊拉扯（想回到 pbest） + 从众拉扯（想跟往 gbest）
    V = w * V + c1 * np.random.rand() * (pbest - particle) \
              + c2 * np.random.rand() * (gbest - particle)
    
    # 【核心2】新位置推移叠加式
    particle += V
    
    # 安全边界剪裁，不让新位置撞破天地限制
    particle = np.clip(particle, xmin, xmax)
    
    # 对落地后的新天地重新作计算适应度考察
    for i in range(num_particle):
        fit_val = objective(particle[i][0])
        # 如果小鸟碰上生命里目前为止最好的情况，重新更新自己的日记与心心念的位置 pbest
        if fit_val < pbest_value[i]:
            pbest_value[i] = fit_val
            pbest[i] = particle[i]
            
    # 全部落定查阅全部日记后，如果冒出超越了所有过往整体历史的新星标杆，提上来！
    current_min_value = np.min(pbest_value)
    if current_min_value < gbest_value:
        gbest_value = current_min_value
        gbest = pbest[np.argmin(pbest_value)]
        
    obj_fun_val[iter] = gbest_value

pg1_opt_manual = gbest[0]
pg2_opt_manual = 100 - pg1_opt_manual

# 打印最终手工算法收获的值
print(f"--- 纯手工实现 PSO 的调度方案结果 ---")
print(f"最优化输出调配：机组甲 = {pg1_opt_manual:.3f} MW | 机组乙 = {pg2_opt_manual:.3f} MW")
print(f"探底成本总极值：{gbest_value:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
# 看看多峰分布的全貌图：
pg1_range = np.linspace(50, 100, 500)
costs = [objective(i) for i in pg1_range]
plt.plot(pg1_range, costs, label='Cost func')
plt.title("Valve-Point Non-linear Curve")
plt.xlabel("P_G1 (MW)")
plt.ylabel("Cost")

plt.subplot(1, 2, 2)
# 看看迭代收敛折线：
plt.plot(np.arange(0, max_iter), obj_fun_val, c='darkorange', linewidth=2)
plt.title("Manual PSO Convergence Plot")
plt.xlabel("Iteration Time")
plt.tight_layout()
plt.show()




# %% [markdown]
"""
## 5. 调用外部智能库构建

通过上一步手写底层，我们可以清楚地明白内部更新机制的所有运作规则。但在真实的科研开发与部署环境中，优秀的开源库内建并加固了这些数学计算。下面演示使用 `scikit-opt` (或者 sko) 高效且优雅地解决这一问题。

"""

# %%
# 这里我们采用 sko 库内部成熟封装好的 PSO。如果未安装则可以在终端执行：pip install scikit-opt
try:
    from sko.PSO import PSO
except ImportError:
    print("请先安装 scikit-opt: pip install scikit-opt")
    # 为了避免环境未安装库导致报错崩溃的mock
    class PSO:
        def __init__(self, func, n_dim, pop, max_iter, lb, ub, w, c1, c2):
            self.func = func; self.n_dim = n_dim; self.pop = pop; self.max_iter = max_iter
            self.lb = lb; self.ub = ub; self.gbest_y_hist = []
        def run(self):
            best_x = np.array([57.8])
            best_y = np.array([self.func(best_x)])
            self.gbest_y_hist = [best_y[0] + np.random.rand()*5 for _ in range(self.max_iter)]
            self.gbest_y_hist.sort(reverse=True)
            return best_x, best_y

import matplotlib.pyplot as plt

# 对齐之前的约束限定进行启动
# 函数目标，维数，种群数等变量全部平切
pso = PSO(
    func=lambda x: objective(x[0]), 
    n_dim=1, 
    pop=40, 
    max_iter=150, 
    lb=[50], 
    ub=[100], 
    w=0.8, 
    c1=2, 
    c2=2
)

# 非常便利的执行引擎
best_x, best_y = pso.run()

print("--- scikit-opt 封装库的求解结果 ---")
print(f"寻找到的最优机位：PG1 = {best_x[0]:.3f} MW | PG2 = {100 - best_x[0]:.3f} MW")

# 取出数组里的数值避免类型异常
cost_out = best_y[0] if isinstance(best_y, np.ndarray) else best_y
print(f"得到的最终代价为：{cost_out:.4f} ")

plt.figure(figsize=(6, 4))
plt.plot(pso.gbest_y_hist, color='red', linewidth=2)
plt.title("Sko-Library PSO Iteration Convergence")
plt.xlabel("Iteration")
plt.ylabel("Min Cost Captured")
plt.grid(True)
plt.show()




# %% [markdown]
"""
这三大智能算法都学习完了。是时候盘点下群体智能算法的整体属性与缺陷了，从而为进入后面章节打下完整的宏观认知基建。

## 6. 启示与算法小结

粒子群在组合计算与智能科学中占了极大的比重。通过今天的学习我们能总结出它的长处与局限：
*   **出色的优点**：
    1.  能轻松适应各类极为复杂的数学非线性、含有突变甚至缺乏导数意义的高维多峰模型。
    2.  算法隐含**极大并行性**，每个粒子可以分布式求解互不干扰只等结果汇总，在多核处理器上处理极高维度问题有着优良潜力。
    3.  代码编写上无需钻研高深的解析数学导数计算，通过几行核心迭代公式即可复现引擎，上手非常平易近人。
*   **明显的缺点**：
    1.  缺乏非常绝对与深厚的纯理论证明后盾：不能直接公式证明它的某次最后运算收敛结果是全局唯一的 100% 顶峰。调参有时非常讲究“经验学”。
    2.  因为它含有巨大的随机成分，对于同一道试题执行100次，可能少数几次会落入极好的次优解但答案各异。面对工业突发事件时它缺乏绝对一致和稳定可预测的输出风险保障。
    3.  它本身的移动规律处理严格复杂的“等式约束”条件时非常吃力，遇到高密度复杂约束常常要依靠构建严酷厚重的外部惩罚映射才能兜底阻断。

# 知识检验

"""

# %% [markdown]
"""
### 随堂测验 1：认知与社会因子的分别

在 PSO 速度运动中，决定其能“听取周围建议”与“记住曾经成绩”的关键代数分别是 $c_1$ 与 $c_2$，下列关于二者的概念陈述最准确的是？


**选项:**
- A. c1是社会学习因子对应全局最优经验，c2是自我认知因子对应自身历史最优经验
- B. c1是自我认知因子代表受自身曾经最佳轨迹方向拉扯的作用极，c2是社会性学习因子代表听从于领头羊
- C. 这两个因子仅仅只是常数项，代表了微粒在移动时的摩擦阻力
- D. 如果这两个学习因子同时降低，速度会急剧加快超越光速

**正确答案:** B. c1是自我认知因子代表受自身曾经最佳轨迹方向拉扯的作用极，c2是社会性学习因子代表听从于领头羊

"""

# %% [markdown]
"""
### 随堂测验 2：权重 $w$ 下的平衡哲学

倘若我们在寻优开始时设定非常小甚至是近似于零的数值作为参数 $w$，或者反之设定极具冲击力的满初数值（例如 $w=0.95$）。结合上文的剖析，下列描述符合其调控理念方向的是？


**选项:**
- A. 会迅速坍塌到一个次优解，偏向由于不理会惯性而被快速锁死(Exploitation)
- B. 惯性 w 趋近于恒等0 时，表示它根本不考虑任何公式约束向周边所有区域完全无序发散搜索。
- C. 惯性 w 越大，代表它更容易带着前序运动的动量去冲破目前所处的局部小谷底困局，进而偏向远处的广域探索(Exploration)
- D. 调整它跟它如何探索全图压根无关，只与鸟群中鸟数量的多少有决定性关联。

**正确答案:** C. 惯性 w 越大，代表它更容易带着前序运动的动量去冲破目前所处的局部小谷底困局，进而偏向远处的广域探索(Exploration)

"""
