# %% [markdown]
"""
# 1. 为什么要选取或提取特征？

![降维与特征压缩](images/2020-04-12-10-26-41.png)

对于机器学习来说，最重要的就是数据，数据中最重要的就是特征（Features）。如果你的数据中有充满干扰且毫无规律的信息（噪声），那么训练出来的模型也会像一团乱麻（Garbage in, garbage out）。
因此在将数据送入模型前，经常需要处理一下原始特征，留下有用的或制造更有效的新特征。

特征处理的主要目的包括：
1. **减少计算复杂度**：让模型训练更快。
2. **降低过拟合的风险**：减少多余特征造成的噪声，提高泛化能力。
3. **引入领域知识**：将数据整理成更容易被模型发现规律的形式。
4. **提升模型解释性**：少量直接影响结果的特征，更容易讲出“好故事”。

应对特征太麻烦的问题，我们主要有两大路线：**特征选取（Feature Selection，直接扔掉不要的）** 和 **特征提取（Feature Extraction，把多个特征融合成新的几个特征）**。

## 2. 特征选取 (Feature Selection)

特征选取顾名思义：挑好的，扔坏的。通常分为两大类方法：Filter 和 Wrapper。

### 2.1 过滤法 (Filter Method)
![Filter Method 示意图](images/2020-04-03-01-01-19.png)

这种方法简单粗暴，凭借着统计学或者相关系数，在把数据给模型之前就把特征过滤一遍。
比如两个连续变量，我们可以算皮尔逊相关系数：若特征与标签极度相关，留；若极度不相关，抛。
若遇到分类问题，就可以用方差分析（ANOVA）验证该特征是否影响分类结果。

当然，除了数学计算，在 EDA（探索性数据分析）中通过画图观察，也是一种超级直接的“Filter”手段。
比如在上一节 Iris 的对角散点图中：

有些散点图（例如花瓣长、宽的关系）分界极其明显，哪怕只保留一个维度，都能把品种分清。

### 2.2 包装法 (Wrapper Methods)

包装法就是暴力测试——训练模型，看效果，好就留，差就走。

![Wrapper 流程](images/2020-04-03-13-52-59.png)

通常有两个典型的套路：
1. **前向选择 (Forward Selection)**:
   从只用 1 个特征开始测试所有特征，挑出单兵作战最好的。然后再试用 2 个特征（留下最强单兵+其余轮换测试），找出最佳的双核心组合。以此类推，直到模型精度不再提升。
   ![前向选择](images/2020-04-03-13-51-48.png)
   
2. **后向剔除 (Backward Elimination)**:
   满配首发，用所有特征训练，然后每次扔掉一个，看看效果是不是断崖式下跌。如果某特征被扔掉居然丝毫不影响表现，那就彻底把它剔除。
   ![后向剔除](images/2020-04-03-13-51-57.png)

---

## 3. 特征提取与主成分分析 (Principal Component Analysis, PCA)

假设你有一项考试包括 5 科（语文成绩、文综成绩、写作水平；物理成绩、数学成绩）。如果仅仅靠上面说的 Filter 去除，你可能会损失掉某些细微的区分维度。
但实际上，前三科高度相关且受“文科思维”影响主导，后两科受“理科思维”影响主导。
**特征提取**的思想就是：我们能不能不直接“扔”特征，而是把高度重合的几个旧维度，压缩成一个新维度（例如 $PC_1$: 综合文科能力, $PC_2$: 综合理科能力）？

**主成分分析法（PCA）** 就是最经典的线性降维、特征压缩技术。

### PCA 的核心原理：
PCA 寻找的是原始数据中**变化最大（方差最大）、包含信息最多**的方向。
1. 第一步：找一个能让目前全部数据“撒得最开”的直线（超平面上的向量），这个方向包含了数据最多的差异性，我们把它叫做**第一主成分 (Principal Component 1, $PC_1$)**。
2. 第二步：在保证垂直（正交）于第一主成分的所有方向里，寻找到下一个让数据散得最开的方向，叫做**第二主成分 ($PC_2$)**。
3. 依此类推。如果你本来有 $N$ 个特征，最多能找出 $N$ 个主成分。但是通常绝大部的方差（信息量）都集中在了前几个主成分里。此时只要舍弃那些方差很小的末尾主成分，我们就成功实现了**降维**，同时只丢弃了极少的信息！

让我们到代码里看看这到底是怎么回事：

"""

# %%
import pandas as pd
from sklearn.decomposition import PCA

# 导入数据
iris = pd.read_csv('../../data/07_unit_classification_clustering/iris.csv')
iris.drop(columns=['Id'], inplace=True)
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# 我们尝试直接找4个（最大的）主成分建立 PCA
pca = PCA(n_components=4)
pca.fit(X.values)




# %% [markdown]
"""
我们分析一下跑出来的结果。

"""

# %%
print('各主成分的组成权重 (Components Matrix): ')
print(pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(4)]).round(3))
print('\n各主成分的贡献率 (Explained Variance Ratio): ')
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1} 解释了数据中 {ratio*100:.2f}% 的差异")




# %% [markdown]
"""
惊人的结果！
仅仅依靠**第一个主成分（PC1）**就已经包含了整个 4 维特征空间里 **92.46%** 的方差变量。
第二主成分包含了 **5.31%**。
也就是说，只要我们保留前两个主成分，就把 4 维空间降到了 **2维空间**，但保留了 **97.77%** 的原始数据信息！

这也说明了 Iris 数据存在着极强的内部相关性（花瓣宽和花瓣长度几乎同比例增长）。

## 4. 利用降维（PCA）加速并执行模型训练

我们从前面的课中借用分治与评估代码，尝试使用 2 维 PCA 去训练 SVM 模型。
注意一个**绝对纪律**：PCA 必须只用 `X_train` 去 `fit()` 寻找方向，再通过 `transform()` 施加于训练集和测试集上。绝对不能用包含测试集的全局数据去拟合 PCA（否则就是数据泄漏！即模型提前偷窥了测试集的特征分布）。

"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 切分原始数据
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 定义 2个维度的 PCA
pca_2d = PCA(n_components=2)

# 只在训练集上寻找投影分布 (fit)
pca_2d.fit(X_train_raw)

# 在两个集合上实施维度转换投影
X_train_pca = pca_2d.transform(X_train_raw)
X_test_pca = pca_2d.transform(X_test_raw)




# %% [markdown]
"""
由于数据已经被降到了极佳的二维，我们可以直接把它画出来了：

"""

# %%
plt.figure(figsize=(8, 6))
# 取出 PC1, PC2 绘图
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette="Set1", s=100)
plt.title("PCA 2D Representation of Iris Training Set")
plt.xlabel("First Principal Component (PC1)")
plt.ylabel("Second Principal Component (PC2)")
plt.show()




# %% [markdown]
"""
即便被压扁到了一个平面里，各类数据也基本泾渭分明，分界线非常清晰。
现在，我们把这2个处理好的新特征送给我们的 SVM 进行分类并评估它。

"""

# %%
svm_pca = SVC(kernel='rbf')
svm_pca.fit(X_train_pca, y_train)

# 用测试集来预测
y_pred = svm_pca.predict(X_test_pca)

accuracy = svm_pca.score(X_test_pca, y_test)
print(f"PCA-Reduced SVM Accuracy: {accuracy*100:.2f}%")




# %% [markdown]
"""
即便抛弃了 2 个特征维度，我们依然得到了接近完美的准确率。这正是 PCA 强大的去除冗余噪声、提取核心骨干的能力展示。

"""

# %%
# 可视化最终的混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=svm_pca.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_pca.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix: SVM on PCA 2D Features")
plt.show()




# %% [markdown]
"""
关于主成分分析（PCA）降维带来的效果，以下描述错误的是？


**选项:**
- 可以减少训练模型所需的计算资源。
- 可以完全保留数据集 100% 的原始信息。
- 往往可以消除因为特征冗余带来的影响。

**正确答案:** 可以完全保留数据集 100% 的原始信息。
**提示:** 降维一定会损失一部分特征方差（由于抛弃了尾部的成分），只是对于 PCA，它能最大程度地把信息集中在头部。

"""

# %% [markdown]
"""
我们在对机器学习的数据集实施 PCA 变换时，防止‘数据泄漏（Data Leakage）’的正确做法是？


**选项:**
- 在整个数据集（包含测试集与训练集）上寻找主成分，然后再进行 Train / Test 分割。
- 先进行 Train / Test 分割，只使用 Train 数据集去 fit(拟合) PCA 对象，再用它来 transform 训练与测试集。

**正确答案:** 先进行 Train / Test 分割，只使用 Train 数据集去 fit(拟合) PCA 对象，再用它来 transform 训练与测试集。
**提示:** 如果使用整体数据确立了 PCA 平面方向，则模型相当于提前从测试集偷取了分布信息。

"""
