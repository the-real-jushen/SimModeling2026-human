# %% [markdown]
"""
# 分类与模型选择

就像前面说过的，分类（Classification）属于一种监督学习方法，它和回归（Regression）类似。最大的区别就在于，回归预测的是连续的数值，而**分类的输出是离散的、有限的类别**。

二分类（Binary Classification）优化的目标通常不是像回归那样最小化均方误差（MSE），而是最小化分类误差，通常通过“交叉熵（Cross-Entropy）”或者最大化似然概率来实现。

## 1. 分类的基本原理与最大似然估计

分类不仅要给出一个事物属于哪一类，有时候我们希望它给出属于某类的**概率**。
对于观测样本，它属于某个分类的实际情况只有 0（不属于）和 1（属于）。我们可以把每个样本看成一个条件概率分布。模型训练的本质，就是让模型预测的概率分布尽可能接近样本真实的分布。

这个过程往往使用了**最大似然估计（Maximum Likelihood Estimation, MLE）** 的思想。
假设样本是由某个概率分布产生的，我们的模型含有一组参数 $\theta$。我们想要调整这些参数，使得**“模型产生这些观测已知样本的概率（似然）”最大化**：
$$
\hat{\theta}=\arg\max_{\theta} L(X|\theta)=\arg\max_{\theta} \sum_{i=1}^{n}\log P(x_i|\theta)
$$
使得这种似然函数最大化，就是大多数分类模型（如逻辑回归、深度学习分类器）最底层的训练逻辑。

## 2. 分类器的基本输出与 One-Hot 编码

回归的输出很简单就是一个预测的数字，那么分类模型是如何输出类别的呢？

对于**二分类**（如判断一张照片是猫还是狗），模型通常输出一个属于正类的标量概率。比如在逻辑回归或者支持向量机（SVM）中，通过一个阈值（例如0.5 或者 大于0）来判断到底属于哪一类。

对于**多分类**（例如区分猫、狗、猪三类），模型一般输出的是该样本属于各个类的概率数组，如 `[0.1, 0.1, 0.8]`。
这时候引入一种重要的标签编码方式——**One-Hot 编码（独热编码）**。
例如，类别是“猪”，在只有三种动物分类的情况下，它不被表示为一个数字“3”，而是表示为一个全0但对应位置为1的向量：`[0, 0, 1]`。
预测数组 `[0.1, 0.1, 0.8]` 代表模型认为这个样本有 10% 的可能是一类，10% 的可能是二类，80% 的可能是三类。加起来为 1（满足概率分布）。
我们最终的分类预测，就是取这个数组中最大值的索引（`argmax`）：
$$
\text{预测类别} = \arg\max([0.1, 0.1, 0.8]) = 2
$$

### 如何用二分类模型处理多分类问题？
我们介绍了逻辑回归、基本SVM都是二分类模型。面对多分类，我们有以下两种策略：
1. **One-vs-Rest (OVR)**: 假设有A、B、C三类，则训练 3 个二分类器：一个是“A与非A”，一个是“B与非B”，以此类推。预测时，看哪个模型给出的“属于它自己类别”的概率最高。缺点是如果类别多，非本类别的数据量远大于本类别（数据不平衡）。
2. **One-vs-One (OVO)**: 训练各个类别的两两组合，即 $C_n^2$ 个分类器。然后让他们投票，得票最多的类别赢。缺点是分类器数量较多。

## 3. 常见分类模型的原理介绍

### 3.1 Logistic Regression（逻辑回归）
不要被名字欺骗，这其实是一个**分类**算法！
它是一个比较一般化的线性模型演变而来，用 $\text{sigmoid}$ 函数把实数域的线性输出映射到了 $(0, 1)$ 之间，刚好代表概率。
它使用前面提到的最大似然估计来更新参数，比较适合特征全为连续变量且数量不多、线性可分的数据集。

### 3.2 支持向量机（SVM）
SVM的核心思想是在特征空间中寻找一个能把两类样本分开的**完美超平面**。
而且它不仅仅想分开，还要找到“最宽的马路”——即距离超平面最近的那些“支持向量（Support Vectors）”，到这个平面的距离要**最大化**。
如果数据本身不是线性可分的，SVM 可以通过**核函数（Kernel）**将数据映射到高维空间，然后在高维中划定平面。
SVM 是一种低偏差（Low Bias）模型，非线性拟合能力强，但要注意防止过拟合。

![SVM的基本原理](images/2020-03-04-19-54-20.png)
![SVM中的高维映射](images/2020-03-04-19-58-10.png)

### 3.3 K-Nearest Neighbors (KNN)
物以类聚，人以群分。
KNN 是一种基于距离的非参数分类器。它的规则非常简单暴力：对于一个未知样本，找到其周围距离最近的 K 个已知样本，这 K 个样本中哪一种类别最多，未知样本就归为哪一类。
超参数 $K$ 的选择极其重要。如果 $K$ 太小容易受噪声影响，如果太大类别边界就变得模糊。

![KNN聚类原理示意](images/2020-03-04-20-16-37.png)

### 3.4 决策树（Decision Tree）与随机森林（Random Forest）
像猜谜游戏二十问一样，寻找条件阈值把数据一直往下分，例如“花瓣长度是否大于 2.5cm？” 是则走左分支，否则走右分支。可解释性极强，但非常容易过拟合。

![决策树示意](images/2020-03-04-20-24-29.png)

为了防止决策树的过拟合，我们用**多棵树**组成**森林（Random Forest）**，运用 Ensemble（集成学习）的思想，每棵树看数据的一部分并做出预测，最后集体投票决定分类结果。

![随机森林特征](images/2020-04-02-22-53-19.png)

---

## 4. Iris 数据集实战与数据探索（EDA）
最好的办法就是把刚才提到的模型都跑一遍，看看哪个效果最好。我们来使用经典的 Iris（鸢尾花）数据集。它通过花瓣、花萼的长宽来给花分类（共3类）。

![Iris数据特征1](images/2020-03-03-12-53-03.png)
![Iris数据特征2](images/2020-03-03-12-55-55.png)

"""

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据并去掉ID列（ID对分类毫无帮助）
iris = pd.read_csv('../../data/07_unit_classification_clustering/iris.csv')
iris.drop(columns=['Id'], inplace=True)
iris.head()




# %% [markdown]
"""
由于特征有四个（4维空间），我们人类只能直观看到2维或者3维。我们可以通过绘制特征两两组合的散点图，来观察各类花之间的分离情况。
用 Pandas 内置的强大的 `pairplot`（Seaborn底层）来全景观察特征之间的相互覆盖情况：

"""

# %%
sns.pairplot(iris, hue='Species', diag_kind='hist', palette='Set1')
plt.show()




# %% [markdown]
"""
从散点矩阵图中我们可以看出：`Iris-setosa` （红色）与另外两类区分得非常明显，而 `versicolor` 和 `virginica` 在有些特征组合下有轻微重叠，但也存在比较清晰的边界。这说明这组数据**非常好分类**。

## 5. 模型切分与训练 (Train-Test Split)
把数据集切分为训练集和测试集是必须的步骤，保证模型在没有看过的数据上进行测评（泛化能力验证）。

"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 获取特征 X 和标签 y
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# 拆分训练与测试集（50%作测试），并固定随机种子
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 建立一个字典存储要训练的模型
models = {
    'SVM (RBF Kernel)': SVC(kernel='rbf'),
    'KNeighbor (K=5)': KNeighborsClassifier(n_neighbors=5),
    'RandomForest (100 Trees)': RandomForestClassifier(n_estimators=100)
}

# 循环进行模型训练并做基础的得分输出
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model: {name} | Test Accuracy: {acc:.4f}")




# %% [markdown]
"""
## 6. 模型效果评估与混淆矩阵 (Confusion Matrix)

在分类问题中，只看**准确率 (Accuracy)**（即分类正确的样本占总样本的比例）往往是不够的，尤其是在**数据类别不平衡**的情况下。
举个极端的例子：假设100个病人中只有2个是绝症（阳性），如果我们开发一个“永远猜测没病（阴性）”的无脑模型，它的准确率高达98%，但这模型毫无意义，因为它连一个真正的病人都没找出来！

因此，我们需要引入**混淆矩阵 (Confusion Matrix)**。对于二分类问题，它将分类结果划分为四种情况：
- **真正例 (True Positive, TP)**：实际是正例，预测也是正例（如：真的是病人，且被诊断出有病）。
- **假阴性 (False Negative, FN)**：实际是正例，却预测为负例（如：真的是病人，却被漏诊了）。
- **假正例 (False Positive, FP)**：实际是负例，却预测为正例（如：健康人，却被误诊为有病）。
- **真负例 (True Negative, TN)**：实际是负例，预测也是负例（如：健康人，且被诊断为健康）。

基于混淆矩阵，除了准确率之外，我们还能计算出几个非常关键的评估指标：

1. **真正率 (TPR) / 召回率 (Recall) / 灵敏度 (Sensitivity)**:
   - **公式**：$TPR = \frac{TP}{TP + FN}$
   - **含义**：在所有**实际为正**的样本中，模型成功找出了多少。
   - **实际例子**：在**癌症筛查**中，我们最看重召回率。因为漏诊（假阴性FN）的代价是病人的生命，宁可错杀一千（高误诊），不可放过一个（低漏诊），所以要求极高的 TPR。

2. **假正率 (FPR)**:
   - **公式**：$FPR = \frac{FP}{FP + TN}$
   - **含义**：在所有**实际为负**的样本中，模型**错误地报警**了多少。
   - **实际例子**：在**垃圾邮件过滤**中，正常邮件是负例，垃圾邮件是正例。假正例FP意味着一封正常甚至重要的工作邮件被丢进了垃圾箱，这会让用户非常愤怒。因此，在这个场景下，我们会严控 FPR（即宁可多漏掉几封垃圾邮件，也不能把正常邮件错杀）。

3. **精确率 (Precision) / 查准率**:
   - **公式**：$Precision = \frac{TP}{TP + FP}$
   - **含义**：在模型**所有预测为正**的样本中，有多少是真的正例。也就是说，模型的“报警”有多准。

4. **F1 分数 (F1-Score)**:
   - **公式**：$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
   - **含义**：精确率和召回率往往是矛盾的（想找得全，就容易找得不准；想找得准，就会漏掉一些）。F1 分数是两者的**调和平均数**，用于综合衡量模型在正例上的表现，是一个非常重要的单值综合考核指标。

### 关于多分类的评估指标
上面的指标都是针对“二分类”定义的。当我们在做**多分类**（如 Iris 数据的三种花）时，该如何评价呢？
通常的做法是使用 **One-vs-Rest (OvR)** 策略，把每一个特定类别当成“正例”，把其他所有类别当成“负例”，这样就可以为每一个类别都计算出一个局部的 Precision、Recall 和 F1-score。
最后，根据需求将这些局部指标进行汇总，常见的方法有：
- **宏平均 (Macro-average)**：不管各个类别样本数量多少，直接把所有类别的指标加起来求算术平均，它对长尾分布（样本极少的）少数类给予了和多数类同等的重视度。
- **微平均 (Micro-average)**：将所有类别的 TP、FP、FN、TN 统一融合在一起，然后计算一个总指标，这在分类严重不平衡时，容易被占主导地位的大类影响。

"""

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def evaluate_and_plot(model, X_train, y_train, X_test, y_test, name):
    # 模型训练与预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*40}")
    print(f"【详细评估】: {name}")
    print(f"{'='*40}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

    # 打印各分类性能指标
    for i in range(len(model.classes_)):
        tp = cm[i, i]
        # 假阳是该列的和减去对角线
        fp = cm[:, i].sum() - cm[i, i]
        # 假阴是该行的和减去对角线
        fn = cm[i, :].sum() - cm[i, i]
        # 真阴是全部样本数减去其他项
        tn = cm.sum() - tp - fp - fn
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"类 [{model.classes_[i]}] : TPR(召回率)={tpr:.2f}, FPR={fpr:.2f}")

# 重新评估各个模型
for name, model in models.items():
    evaluate_and_plot(model, X_train, y_train, X_test, y_test, name)




# %% [markdown]
"""
在混淆矩阵中，如果我们希望考察模型在所有**真实属于类别A**的数据中，成功找出了多少（不漏掉任何一个），我们应该关注哪个指标？


**选项:**
- 准确率 (Accuracy)
- 真正率 / 召回率 (TPR / Recall)
- 假正率 (FPR)

**正确答案:** 真正率 / 召回率 (TPR / Recall)
**提示:** 真正率（也称召回率）专门用来衡量预测出的正例占实际正例样本总数的比例，能够体现不漏掉正例的能力。

"""

# %% [markdown]
"""
如果想要使用支持向量机（SVM）处理线性不可分的分类问题，通常我们需要使用哪种技术？


**选项:**
- 交叉熵 (Cross-Entropy)
- 核函数 (Kernel)
- 独热编码 (One-Hot Encoding)

**正确答案:** 核函数 (Kernel)
**提示:** 可以将样本从低维特征空间映射到高维特征空间，从而实现线性可分的技术。

"""
