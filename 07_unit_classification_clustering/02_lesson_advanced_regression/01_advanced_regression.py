# %% [markdown]
"""
# 1. 更多高级回归模型探索

之前的单元中我们深度学习了线性回归（Linear Regression）体系。但当数据本身具有大量非线性的复杂特征时，传统的线性模型往往无法给出很好的结果。本节课我们将引入 `scikit-learn` 中几种更高级的回归武器，并通过预测**金县(King County)房价**来验证它们的性能。

在正式建模前，我们需要重点学习两个极其重要的概念：**训练/测试集切分**和**数据归一化**。

## 1.1 数据导入

我们要使用一个包含了各种房屋特征（如卧室数、平米数、楼层等）的数据集，来预测房屋的售价 `price`。

"""

# %%
import os
import pandas as pd

# 获取当前脚本所在目录进行相对路径匹配
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
data_path = os.path.join(current_dir, '..', '..', '..', 'data', '07_unit_classification_clustering', '../../data/07_unit_classification_clustering/kc_house_data.csv')
if not os.path.exists(data_path):
    data_path = '../../data/07_unit_classification_clustering/kc_house_data.csv'

# 导入数据并丢掉无效列与空值
data = pd.read_csv(data_path).iloc[:, 2:]  # 舍弃前两列(ID和日期等非数值强相关特征)
data.dropna(inplace=True)

print("--- 房屋数据前 5 行 ---")
print(data.head())

y = data['price'].values
# 丢掉 price，其余都作为特征 X
X = data.drop(columns=['price']).values




# %% [markdown]
"""
## 1.2 关键前置步骤：拆分数据集 (Train Test Split)

在真实的工程中，如果你把你所有的样本数据全都一股脑喂给模型进行训练，模型很可能会产生一种叫**“过拟合(Overfitting)”**的问题——它把这些特定数据的特点甚至噪声死死背了下来，考了100分。但一旦部署去预测完全没见过的新房屋时，它就彻底傻眼了。

因此，为了公正评估模型的泛化能力，我们必须保留一部分数据“不给模型看”，用来充当最后的期末考试题。
通常，我们将数据集按比例（比如 70% vs 30% 或者 75% vs 25%）划分为：
1. **训练集 (Train Set)**：模型用来阅读、学习参数的数据。
2. **测试集 (Test Set)**：模型根本没见过，只用来评估 $R^2$ 得分的全新数据。

"""

# %%
from sklearn.model_selection import train_test_split

# 使用 train_test_split 进行拆分，如果没有指定 test_size，默认切分出 25% 做测试。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"原始数据集样本总数: {len(y)}")
print(f"拆分后，训练集(Train)样本数: {len(y_train)}")
print(f"拆分后，测试集(Test)样本数: {len(y_test)}")




# %% [markdown]
"""
## 1.3 实例化模型并进行基准对比

我们导入 4 种有代表性的模型进行横向对比，让我们先简要了解一下它们的原理特点：

1. **LinearRegression (线性回归)**: 我们最熟悉的基准模型 (Baseline)。它试图在空间中画出一条笔直的线（或建立一个完美的切面平面）来穿过特征点。面对复杂的现实环境非线性波动，往往能力受限。
2. **SVR (Support Vector Regression, 支持向量机回归)**: 大名鼎鼎的支持向量机。SVM 的原始思想是找到一个能划清界限的“超平面”。用在回归预测里时，它允许一部分误差存在，它最巧妙的地方在于能利用各种“核函数 (Kernel)”把低维度难以用直线拟合的数据，强行投射到无穷高维去再做线性拟合。这赋予了其极强逼近高度非线性数据的能力。
3. **KNeighborsRegressor (K-近邻回归)**: 直觉最简单的模型分支。在预测一套新房子时，它的做法是“在以前的数据库里，寻找和当前特征最相似的 K 个房子兄弟”，然后把这几个兄弟当年的历史成交房价加起来“求个平均值”当做这套新房屋的预测价。它高度依赖于特征空间的“距离计算”。
4. **RandomForestRegressor (随机森林回归)**: 非常强大的集成学习“大杀器”。它里面包含了千千万万棵弱小的“决策树”。这些树在特征和数据选取上各不相同，各自得出一个房价预测值，最后“民主投票集结”计算出一个平稳结果。树模型的一大先天优势在于它们**依靠按特征范围分裂节点来做决定（比如卧室>3则归此类），而毫不关心空间的距离长短**。

现在我们开始跑分：

"""

# %%
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# 我们将模型放进字典，方便批量训练和输出
models = {
    'Linear': LinearRegression(),
    'SVR': SVR(),
    'KNR': KNeighborsRegressor(),
    'RFR': RandomForestRegressor()
}

print("--- 基础模型跑分 ($R^2$ 成绩) ---")
predictions = dict()
scores = dict()

for name, model in models.items():
    # 1. 仅用训练集做拟合
    model.fit(X_train, y_train)
    # 2. 在未知测试集上进行预测
    predictions[name] = model.predict(X_test)
    # 3. 输出模型得分 R-squared
    scores[name] = model.score(X_test, y_test)
    print(f'The R-squared of model {name} is: {scores[name]:.4f}')




# %% [markdown]
"""
结果非常震撼：
除了集成学习大杀器 `RandomForest (RFR)` 取得了 0.81 左右的优异成绩外，其他的像 `SVR` 和 `KNR` 表现得简直稀烂，尤其是 `SVR`，它的 $R^2$ 竟然变成了**负数**！（意味着它比你瞎猜全是平均价还要差）。

为什么在理论中非常强悍的算法这里会集体失效呢？这就引出了机器建模流里一个必须掌握的知识盲区：**对模型算法底层的知己知彼**与**数据归一化（标准化）**。

## 1.4 数据归一化 (StandardScaler)

在我们拿到的原始房屋数据中，存在**量纲（数值跨度）极为悬殊**的特征冲突。
比如“房屋面积(sqft)”字段的数值可能动辄几千（如 2500, 3100），而“卧室数量(bedrooms)”的数值往往只有 2 到 5 左右。

- 对于像 `KNR` 和 `SVR` 这样**极度依赖“空间距离”**来计算特征差异的算法来说，数字上相差上千的面积波动会产生巨大的欧式距离，导致模型认为“面积”是世界上唯一重要的事，从而在底层的数学计算上被直接带偏，完全忽略掉了“卧室数”、“卫生间数”这些数值小但极重要特征的敏感度。
- 反观 `RandomForest` 这种**基于树切割分裂节点**工作的算法，它关心的是“这列数字切一刀分走几成”，对尺度的绝对数值大小根本不敏感，所以即使原始数据非常极端，它依然一骑绝尘。

解决距离敏感模型的克星便是：归一化（Standardization），也就是去量纲化。
我们将使用 `scikit-learn` 内置的 `StandardScaler` 来拯救 SVR 和 KNR。这个函数的基本原理是通过公式 $z = (x - u) / s$ 将数据全部平移，强行把各列维度的均值缩放为 0，方差缩放为 1。**这绝不会破坏数据固有的特征分布关系，但却能在一个水平的起跑线上，将不同尺度的属性拉回平起平坐的考察位置。**

"""

# %%
from sklearn.preprocessing import StandardScaler

# 初始化特征缩放器
scaler = StandardScaler()

# 使用训练集的数据去“学习”缩放比例 (一定不能用测试集去学！)
scaler.fit(X_train)

# 对训练集和测试集共同实施同样的比例缩放
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

print("--- 使用 StandardScaler (仅对 X 归一化) 后的跑分 ---")
for name, model in models.items():
    model.fit(X_train_scale, y_train)
    scores['scaled_X_' + name] = model.score(X_test_scale, y_test)
    print(f'The R-squared of model scaled X {name} is: {scores["scaled_X_" + name]:.4f}')




# %% [markdown]
"""
注意观察提升：`KNeighborsRegressor (KNR)` 从刚才的不到 0.5 瞬间越级升到了 0.69，说明特征的同等量纲极大帮助了算法！
但 `SVR` 似乎还是水土不服，我们再把因变量 `y` 也执行相同的归一化试试。

"""

# %%
y_scaler = StandardScaler()
# reshape 是因为 scaler 期望的是 2D 矩阵
y_scaler.fit(y_train.reshape(-1, 1))

y_train_scale = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
y_test_scale = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

print("--- 使用 StandardScaler (对 X 和 y 共同归一化) 后的跑分 ---")
for name, model in models.items():
    model.fit(X_train_scale, y_train_scale)
    scores['scaled_xy_' + name] = model.score(X_test_scale, y_test_scale)
    print(f'The R-squared of model scaled XY {name} is: {scores["scaled_xy_" + name]:.4f}')




# %% [markdown]
"""
现在，`SVR` 也终于恢复了正常的正数评分水平！这个过程告诉我们，**面对依靠空间距离工作的模型算法，进行特征量纲标准化绝对是一切的核心前提**。同时，这也说明不同算法对于超参数与数据环境存在巨大的敏感差异。

随着对 `RandomForest` 和后续模型的进一步超参数调优（修改树的深度、调整学习率等），预测表现还会进一步飞跃。

"""

# %% [markdown]
"""
## 2. 小测验任务：搭建一个 Decision Tree 回归管道

请在此练习中探索 scikit-learn 中尚未用过的决策树模型，根据提示拆分测试集，并给出在测试集上的 R-squared 评估。

"""

# %%
# 在这个练习中，请尝试导入一种之前没有展示过的模型：DecisionTreeRegressor (决策树回归)
import pandas as pd
from sklearn.model_selection import train_test_split
# TODO: 从 sklearn.tree 导入 DecisionTreeRegressor


# 1. 简单造一组用于回归的特征 X 和 目标 y
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=500, n_features=3, noise=0.1, random_state=42)

# 2. TODO: 进行数据集拆分，将 20% 作为测试集
X_train, X_test, y_train, y_test = 

# 3. TODO: 实例化一个决策树回归模型 (DecisionTreeRegressor)，你还可以试着调整超参数 max_depth=5
tree_model = 

# 4. TODO: 在训练集上拟合模型


# 5. 输出在测试集上的跑分 (score) 
print("Decision Tree R-squared:", tree_model.score(X_test, y_test))
