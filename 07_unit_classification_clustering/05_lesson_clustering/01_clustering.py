# %% [markdown]
"""
# 1. 啥是聚类（Clustering）？它和分类（Classification）有何区别？

![聚类导图标题](images/2020-03-05-10-58-16.png)

在前两节课里，我们一直在谈论**分类（Classification）**。分类是一种**监督学习（Supervised Learning）**。它的核心特征是：训练数据中的每一个样本都拥有一个明确的“标签（Label）”，我们提前确切地知道世界上存在哪几类，模型要做的是将新样本对号入座。

而**聚类（Clustering）**不一样，它属于**非监督学习（Unsupervised Learning）**。
在聚类的世界里，我们只有一团散沙般的数据，**完全不知道这些数据属于什么类别、甚至不知道该分成几种类别**。模型没有任何历史标签可以参考（没有 Ground Truth）。
聚类的任务，就是根据数据自身在向量特征空间里的相似程度（大多基于距离），把“长得像”、“靠得近”的数据自动拢成一堆（Cluster）。

> **举个例子**：
> - **分类（Classification）**：淘宝要分辨一个买家是“男性”还是“女性”。“性别”是早已确定的标签，且过去的注册用户填过性别。算法根据过去的有标签数据训练模型，然后预测新用户标签。
> - **聚类（Clustering）**：网易云音乐要开发“私人FM”推荐算法。它并不知道世上有几类听歌群体（摇滚迷？古风迷？ACG迷？都不是），算法仅算出你和某几万名用户的切歌、听歌行为**高度重合距离很近**，就把你和他们“聚成一类”，互相推荐对方喜欢的冷门歌曲。这就叫“物以类聚”。

![空间距离聚类示意](images/2020-03-05-11-03-39.png)

## 2. 聚类的门派 / 算法思想大家族

天下武功出少林，聚类也有多种完全不同的思想衍生出的流派。它们的核心区别在于：
- **怎么定义“同一类”**：靠距离？靠连通性？靠密度？还是靠概率分布？
- **是否需要提前给定簇数**：有些要先给 $K$，有些可以自动发现簇。
- **对噪声和异常点是否鲁棒**：有些算法会把离群点硬分进某类，有些会直接标记噪声。

下面按思路拆开讲，每种都配一个示意图：

### 2.1 Connectivity based (基于连接)
这种方法认为：只要两个点靠得足够近，它们之间就有连线，连在一个大网络里的所有点归为一个簇 (Cluster)；那些相隔太远够不着的孤岛就是别的簇。这种方法一般不需要提前指定有几个类别，而是依靠算法自动得出，你只需要设定“最远连线距离”。层次聚类（Hierarchical clustering）就是这种流派的代表。
![Connectivity base](images/2020-04-03-14-46-05.png)
它的优点是结构可解释性强（可以看到“树状分裂/合并过程”），缺点是当数据量很大时计算成本上升明显。

### 2.2 Density based (基于密度)
顾名思义，这种玩法认为：真正的群体（簇），一定是由一大群足够密集的小伙伴组成的！
你给算法定义一个“侦测半径”和“最少认识的朋友数”。如果某个核心样本点在这半径内找到了足够多的小伙伴，它就牵头成立一个帮派（Cluster）。离所有人远远的孤狼点，统统算作噪声（Noise）。这种流派不需要你提前猜要有几个帮派，且非常擅长聚类形状奇怪的数据。**DBSCAN** 就是一代宗师。
![Density based](images/2020-04-03-14-49-13.png)
它的强项是对噪声点天然友好、对非凸形状（如环形、半月形）效果好；难点在于 `eps` 与 `min_samples` 的选择对结果影响很大。

### 2.3 Distribution based (基于概率分布)
这种流派走的是概率论的优雅路线：它认定同一个簇的点，是由某种隐藏在背后的概率分布（比如二维正态分布）产生的。我们假设空间里有几个这样的分布发生器存在，通过最大化产生这些样本的概率，来把样本归给不同的分布波峰。后面我们将见到的 **高斯混合模型(Gaussian Mixture Model, GMM)** 就是这个流派。
![Distribution based](images/2020-04-03-14-52-07.png)
它的优势是可以输出“属于每个簇的概率”，尤其适合簇之间有重叠、边界模糊的情况；代价是建模和调参通常比 K-Means 更复杂。

### 2.4 Centroid based (基于质心)
最为著名的经典聚类，认为每一个簇都有一个核心地带，称之为中心点（Centroid）。这就是本文要大书特书的 **K-means**。
![Centroid based](images/2020-03-05-11-10-56.png)
它的优点是速度快、实现简单，适合“类内接近球形且规模相近”的数据；缺点是对离群点敏感，而且不擅长处理环形/弯月等复杂拓扑。

---

## 3. K-Means Clustering 的基本原理与工作流程 (Centroid based)

K-Means 的 K 代表你想把数据分成几个簇，Means 指的是每个类别里所有点的平均坐标（即重心/质心）。我们用最直观的步骤拆解它的算法全过程：

![原始数据点](images/2020-03-05-11-04-15.png)
**第一步：** 拍脑袋决定你要把东西分为几类，假设我们想要分为3类（即设定 $K=3$）。
**第二步：** 在整个数据空间里，系统纯随机“空投”三个中心点标记。
![空投中心点](images/2020-03-05-11-07-24.png)

**第三步（划地盘）：** 计算所有所有样本点到这三个随机中心点的直线距离。哪个点离它近，它就打上那一帮的标签！
![初步划归](images/2020-03-05-11-10-56.png)

**第四步（调整中心）：** 目前的地盘划分很不合理，对不对？没关系！现在让每个帮派里的所有点重新投票，把该势力的中心点强行移动到真实小弟们坐标的**正中心（平均位置）**上。
![中心移动](images/2020-03-05-11-13-24.png)

**第五步：** 循环重复上述的“划地盘”、“调中心”动作。一直到中心点的位置彻底不再发生位移（即算法收敛收兵）。
下方动图完美地展示了中心点通过循环不断找准位置的过程：
![K-means演进动图](images/wkmeans_1.gif)

### K-Means 的评价与 K 值的选取门道
我们需要知道分的好不好，就需要一个分数评价机制。对于 K-Means 最经典的指标叫 **Sum of Squared Errors (SSE/Inertia)**，也就是所有点到属于它自己簇的中心点的距离的平方和：
$$
   se=\sum_{k=1}^{Clusters}\sum_{i=1}^{n_{k}}Distance_{ki}
$$

这带来了一个巨大的哲学问题：如何选定合理的 $K$？如果你选 $K=所有人数$，也就是一个人自己就是一个簇，中心就是自己，距离 SSE 彻底掉到了0，但这毫无意义！
如果我们一无所知，我们可以用 **Elbow Method (肘部法则)**！
即我们从 $K=2$ 慢慢往上涨，把每个 $K$ 对应的这 SSE 数字画出折线图。当增加一个 $K$ 能让 SSE 发生断崖下跌，后继却开始变得极其鸡肋平缓时，就像人手臂的“手肘关节”被卡住的那一点，那往往就是绝佳的 $K$。
![拐点法1](images/2020-03-05-12-49-00.png)
在这张图中，明显的拐弯肘部就是我们想要的 $K=3$。
![拐点法2](images/2020-03-05-12-58-10.png)
![拐点法3](images/2020-03-05-12-58-47.png)

好！我们先在熟悉的 Iris 数据上测一回 K-Means与肘部法则！

"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

# 压制无害的 KMeans 内存警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 读入依然是没有标签意识的 Iris，强行切除最后一列品种名，当作纯黑盒数据
iris = pd.read_csv('../../data/07_unit_classification_clustering/iris.csv')
iris.drop(columns=['Id'], inplace=True)
# 假装没看见标签y，只用前四列作为未探索过的特征
X_df = iris.iloc[:, :-1]
X_df.head()




# %% [markdown]
"""
接下来，我们通过代码实施肘部法（Elbow Method）来评估 KMeans 选择不同 K 时的 SSE （sklearn 中叫 `inertia_` 惯性）。

"""

# %%
# 记录不同k值下的se
distortions = []
models = []
K_list = range(1, 10)

for k in K_list:
    # random_state用于固定结果，n_init设置初始化次数（符合最新版sklearn要求）
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_df)
    models.append(model)
    distortions.append(model.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_list, distortions, 'bx-', linewidth=2, markersize=8)
plt.xlabel('设置的 K 类别个数 (k)')
plt.ylabel('总重心距离偏差 (Distortion / Inertia)')
plt.title('通过 Elbow Method (手肘法) 寻找最佳的 K')
plt.grid(True)
plt.show()




# %% [markdown]
"""
果不其然，在 K=3 （或者K=2处也有很强拐点信号，这也是为什么生物学上有时认为Iris两个大类更准确的区别）以后，曲线下滑变得极其鸡肋。由于我们有上帝视角，我们选择看模型在 $K=3$ 时的自动分类情况。
由于有4个维度，我们只挑2个最直观的维度（比如花萼宽度和花瓣长度：列标 1 和 2）来作图，颜色由模型自己计算得来：

"""

# %%
# 选择第3个模型（K=3这个下标在list里是索引 2）
best_model = models[2]
predicted_labels = best_model.predict(X_df)

plt.figure(figsize=(8, 6))
# 选取列索引为1和2的两项特征，画出聚类的散点
plt.scatter(X_df.values[:, 1], X_df.values[:, 2], c=predicted_labels, cmap='viridis', s=80, alpha=0.8)
plt.title("K-Means (K=3) 在 Iris 花萼宽/花瓣长 两维度上的聚类着色")
plt.xlabel("Sepal Width")
plt.ylabel("Petal Length")
plt.show()




# %% [markdown]
"""
## 4. 探究特殊拓扑空间的聚类克星与利器
在上面，K-Means对于“圆团团”的数据游刃有余。可是世界很复杂，如果我们的数据长这样呢？一种是内环包外环（同心圆型形状）！这种基于两圆的分布，单纯用“直线连线质心距离”的 K-Means 就彻底抓瞎了，会把左右各切一半。

此时，**DBSCAN（基于密度的方法）**站了出来。它能顺藤摸瓜探测高密度的连贯区域，完全不管全局核心在哪里，完美破局！

我们先制作假数据：半月与同心圆型。

"""

# %%
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 创造一个大圈套小圈的黑盒数据
N = 300
r_1 = 0.5
r_2 = 5
theta = np.linspace(0, 2*np.pi, N)

# 创造点，稍微加上噪音随机扭动
X_1 = r_1 * np.array([np.cos(theta), np.sin(theta)]) + np.random.rand(2, N)
X_2 = r_2 * np.array([np.cos(theta), np.sin(theta)]) + np.random.rand(2, N)

X_weird = np.append(X_1.transpose(), X_2.transpose(), axis=0)

plt.figure(figsize=(6, 6))
plt.scatter(X_weird[:, 0], X_weird[:, 1], color='gray', s=30)
plt.title("人类一眼就能分出来这是2个圈，机器能吗？")
plt.show()




# %% [markdown]
"""
如果使用基于密度的法师 DBSCAN：
参数解析：
`eps=1`: 你侦查寻找小伙伴探测雷达的半径长度
`min_samples=3`: 找到几个小伙伴才能确认自己混上了一个圈的核心帮派？

和其他方法对比一下：
- **对比 K-Means**：K-Means 依赖“到质心的直线距离”，更偏好圆团状簇；DBSCAN 依赖局部密度连通，能识别环形和不规则形状。
- **对比层次聚类**：层次聚类给你“结构树”，解释性强；DBSCAN 更强调自动识别高密度簇并过滤噪声。
- **对比 GMM**：GMM 假设数据来自若干高斯分布，适合概率建模；DBSCAN 不依赖分布假设，适合几何形状复杂的数据。

"""

# %%
# 应用 DBSCAN 模型
dbscan_model = DBSCAN(eps=1.1, min_samples=3)
dbscan_result = dbscan_model.fit_predict(X_weird)

plt.figure(figsize=(6, 6))
plt.scatter(X_weird[:, 0], X_weird[:, 1], c=dbscan_result, cmap='coolwarm', s=50)
plt.title("DBSCAN 密度聚类的完美分类，它认出了连贯环！")
plt.show()

# 用同一份数据对比 K-Means 与 DBSCAN
kmeans_weird = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_weird_labels = kmeans_weird.fit_predict(X_weird)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_weird[:, 0], X_weird[:, 1], c=kmeans_weird_labels, cmap='viridis', s=35)
axes[0].set_title("K-Means on Ring-shaped Data")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

axes[1].scatter(X_weird[:, 0], X_weird[:, 1], c=dbscan_result, cmap='coolwarm', s=35)
axes[1].set_title("DBSCAN on Ring-shaped Data")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")

plt.suptitle("同一数据集对比：K-Means vs DBSCAN")
plt.tight_layout()
plt.show()




# %% [markdown]
"""
太强了！完全贴合了内圈和外圈环形状，如果是 K-Means 它会把这里一刀劈成左右两圆。 
最后介绍概率派扛把子GMM，适合多个不规则、甚至稍微有点交叉的椭圆形点簇。它本质上是假设数据全都是由不同的正态（高斯）分布叠加混合生成的，它用一种极其复杂的机制（EM算法）找到了这些概率场的大本营在哪里！

和常见算法做个快速对比：
- **对比 K-Means**：K-Means 是“硬分类”（每个点只属于一个簇）；GMM 是“软分类”（每个点可同时对多个簇有概率）。
- **对比 DBSCAN**：DBSCAN 擅长发现任意形状簇并识别噪声；GMM 更擅长描述“有重叠、近似椭圆”的概率团。
- **适用场景差异**：若你更关心“每个点属于各簇的置信度”，GMM 往往更有优势。

下面我们生成两个“有少量重叠”的高斯云团：其中一个分布的方差更大，以模拟现实中“一个人群更分散、另一个人群更集中”的情况。

"""

# %%
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

# 创造两个部分重叠的高斯云：其中一个分布方差更大
# mu_1, sigma_1 定义第一个高斯分布的均值和协方差矩阵（更分散）
mu_1 = [1, 3]
sigma_1 = [[5.0, 1.6], [1.6, 2.5]]
# mu_2, sigma_2 定义第二个高斯分布的均值和协方差矩阵（更集中）
mu_2 = [3, 1.5]
sigma_2 = [[1.2, -0.3], [-0.3, 0.8]]

X_gmm = np.append(
    np.random.multivariate_normal(mu_1, sigma_1, 1000), 
    np.random.multivariate_normal(mu_2, sigma_2, 1000), 
    axis=0
)

# 拟合我们设定的2层高斯分量混合模型
gmm_model = GaussianMixture(n_components=2, random_state=42)
gmm_model.fit(X_gmm)

# 利用生成出的概率模型分布去直接预测数据属于哪层分布
gmm_result = gmm_model.predict(X_gmm)
gmm_prob = gmm_model.predict_proba(X_gmm).max(axis=1)

# 用同一份数据做 K-Means 对照
kmeans_gmm = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_gmm_result = kmeans_gmm.fit_predict(X_gmm)

# 根据概率密度画等高线地形图！
plt.figure(figsize=(8, 6))
mesh_x, mesh_y = np.linspace(-8, 10), np.linspace(-5, 8)
mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
mesh_XX = np.array([mesh_X.ravel(), mesh_Y.ravel()]).T

# 得分取负，以画连续等高线
mesh_Z = -gmm_model.score_samples(mesh_XX)
mesh_Z = mesh_Z.reshape(mesh_X.shape)

# 并排可视化：K-Means 与 GMM
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：K-Means 结果
axes[0].scatter(X_gmm[:, 0], X_gmm[:, 1], c=kmeans_gmm_result, cmap='viridis', s=10, alpha=0.45)
axes[0].set_title("K-Means on Overlapping Gaussian Data")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

# 右图：GMM 结果 + 概率地形 + 置信度
CS = axes[1].contour(
    mesh_X,
    mesh_Y,
    mesh_Z,
    norm=LogNorm(vmin=1, vmax=100),
    levels=np.logspace(0, 1, 10),
    cmap='hsv',
    alpha=0.45,
)
scatter = axes[1].scatter(
    X_gmm[:, 0],
    X_gmm[:, 1],
    c=gmm_result,
    s=10 + 20 * gmm_prob,
    cmap='viridis',
    alpha=0.5,
)
axes[1].set_title("GMM on Overlapping Gaussian Data")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")

fig.colorbar(CS, ax=axes[1], shrink=0.8)
plt.suptitle("同一数据集对比：K-Means vs GMM（右图点大小表示GMM置信度）")
plt.tight_layout()
plt.show()




# %% [markdown]
"""
如果给你几十万用户的购物频次和金额表，里面没有任何人工标注的这类例如“高潜力”、“流失倾向”此类的结论。这在机器学习中属于以下哪一种计算范畴？


**选项:**
- 非监督学习中的聚类 (Clustering) 算法
- 监督学习中的分类 (Classification) 算法
- 监督学习中的回归 (Regression) 算法

**正确答案:** 非监督学习中的聚类 (Clustering) 算法
**提示:** 因为你手中毫无标签（Labels），不知道具体的正确分类有哪些形态。你需要自发性地去挖掘人以群分的数据形态。

"""

# %% [markdown]
"""
在应对类似“环形结构”或者严重不规则人群分布时，哪种算法有天然得天独厚、顺藤摸瓜寻找连通簇的优势？


**选项:**
- K-Means 基于重心的算法
- Logistic Regression 逻辑回归
- DBSCAN 基于密度的聚类法

**正确答案:** DBSCAN 基于密度的聚类法
**提示:** K-Means 是基于重心的直线距离判断算法，如果面对环形或复杂互相包裹的数据会一刀切断导致分类错误，我们需要一种能顺藤摸瓜的算法。

"""
