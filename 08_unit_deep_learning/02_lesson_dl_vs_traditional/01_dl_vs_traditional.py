# %% [markdown]
"""
# 深度学习 vs 传统机器学习：以手写字母识别为例

在前几节课中，我们学习了传统的机器学习算法（如 SVM、逻辑回归等）。在处理复杂数据（例如图像、手写轨迹、语音）时，传统机器学习和深度学习最大的区别在于**特征提取 (Feature Extraction)** 的环节。

- **传统机器学习**：极度依赖人类专家介入，需要人工设计、提取特征（Feature Engineering）。如果我们对数据不了解，就很难提取有效特征。
- **深度学习**：采用**端到端 (End-to-End)** 的理念，我们只需输入极少处理的原始数据，神经网络会自动从海量数据中学习并提取复杂的特征规律。

为了直观体会这一区别，我们来看一个**手写字母识别 (A-Z)** 的真实案例。
注意：这不是一张张完整的图片，而是在数字手写板上画出的**轨迹数据 (Time, X, Y)**。

## 1. 传统机器学习思路：手工提取特征 + SVM

在使用传统机器学习时，人类专家必须先观察笔画数据，推导出几十个可能对分类有帮助的统计学特征。
例如专家设计了 25 个特征：
- 宽高比 (AspectRatio)
- X 和 Y 的平均绝对偏差 (MADX, MADY)
- 速度和方向的变化率 (Derivative)
- 局部极值点的个数 (NumXMin, NumYMax) ...... 等等。

我们可以先来看看，为了得到这些特征，我们需要写多少代码：

"""

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin

# === 传统机器学习繁琐的特征工程代码展示 ===

def preprocess(data):
    # 归一化时间 [0, 1]
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    data['Time'] = data[['Time']].apply(max_min_scaler)
    data['X'] = data['X'] * 1.5 # 修复纵横比
    # 中心化 X & Y
    data['X'] = data['X'] - data['X'].mean(skipna=True)
    data['Y'] = data['Y'] - data['Y'].mean(skipna=True)
    # 缩放至面积为1
    scl = 1/np.sqrt((data['X'].max() - data['X'].min())*(data['Y'].max() - data['Y'].min()))
    data['X'] = data['X'] *scl
    data['Y'] = data['Y'] *scl
    # 计算一阶导数 (速度)
    data['dXdT'] =data['X'].diff()/data['Time'].diff()
    data['dYdT'] =data['Y'].diff()/data['Time'].diff()
    # ... 省略部分平滑代码 ...
    return data

def extractfeatures(data):
    # 计算极值点个数、信号间的相关系数、宽高比等 25 个特征
    df = data.drop('Time',axis=1)
    c = df.corr()
    feat = [
        (data['Y'].max() - data['Y'].min())/(data['X'].max() - data['X'].min()),  # AspectRatio
        data['X'].sub(data['X'].mean()).abs().mean() , # MADX
        # ... 限于篇幅，这里不列出所有几十个统计量，但你需要为每一个特征设计数学公式 ...
    ]
    return feat




# %% [markdown]
"""
因为对几千个样本挨个执行这些数学运算会耗费一定时间（而且代码十分冗长），我们在课程中直接加载**提前用上述代码提取好的 25 个特征**。我们使用支持向量机（SVM）进行分类。

"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1.1 读取专家手工提取的 25 个特征数据（letterdata.csv）
letterdata = pd.read_csv("../../data/08_unit_deep_learning/letterdata.csv", index_col=0)

# 查看特征维度和前两行
print(f"数据总维度: {letterdata.shape}")
display(letterdata.head(2))

# 特征列
X_trad = letterdata.drop(columns='Character')
# 标签列（真实字母）
y_trad = letterdata['Character']

# 划分训练集和测试集
X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(
    X_trad, y_trad, test_size=0.25, random_state=123
)

# 1.2 训练传统的 SVM 分类器
print("\n开始训练基于人工特征的 SVM 模型...")
svm_model = SVC(kernel='rbf', C=10)
svm_model.fit(X_train_trad, y_train_trad)

# 1.3 测试集上的评估
svm_acc = svm_model.score(X_test_trad, y_test_trad)
print(f"传统机器学习 (人工特征 + SVM) 测试集准确率: {svm_acc*100:.2f}%")




# %% [markdown]
"""
我们可以看到，人工绞尽脑汁提取了 25 个特征，SVM 模型获得了不错的准确率。但如果面对更加复杂的三维甚至高维任务，人类就很难想到那么多统计特征了。

## 2. 深度学习思路：原始轨迹输入 + 神经网络层层提取

现在，我们把这套流程换成**深度学习**来做。
我们**不再进行任何复杂的特征工程（不计算方差、不找极值点）**，而是仅仅对原始轨迹数据在时间轴上进行极其简单的差值重采样（插值拉伸）。我们将每个字母的 $X, Y$ 坐标分别插值为均匀的 30 个点。然后首尾相接，拼成一条长度为 $60$ (30个X, 30个Y) 的原始数据向量。

把这最朴素的 60 个数字点喂给**人工神经网络 (Neural Network)**。我们让模型自己去学什么才是关键特征！

"""

# %%
# 我们引用同目录下的 dataLoader.py 脚本帮我们完成轨迹的自动拼接重采样
import dataLoader as dl
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# 导入轨迹文件夹数据为 60 维的原始向量
# features: 样本数量 x 60的矩阵 (30个X坐标 + 30个Y坐标)
features, labels = dl.readData(r"../../data/08_unit_deep_learning/Data_csv")

print(f"深度学习 原始特征矩阵形状: {features.shape}")

# 画出随便一个样本的轨迹，确认一下
sample_idx = 10
plt.figure(figsize=(3,3))
# 绘制出这30个X点和30个Y点的连线
plt.plot(features[sample_idx, 0:30], features[sample_idx, 30:])
plt.title(f"Label: {labels[sample_idx]}")
plt.axis('equal')
plt.show()

# 将原始特征划分测试与训练集
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# 神经网络无法理解类标 'A', 'B' 等英文字符，我们利用 dataLoader 将它们转为数字 (0-25)
y_train_dl_num = np.array([dl.letter2Number(letter) for letter in y_train_dl])
y_test_dl_num = np.array([dl.letter2Number(letter) for letter in y_test_dl])




# %% [markdown]
"""
### 2.2 构建多层全连接神经网络框架 (MLP)

我们的神经网络包含 2 个隐藏层：第一层 128 个神经元，第二层 50 个神经元。

"""

# %%
model = keras.Sequential([
    keras.Input(shape=(60,)), # 输入层接收 60 长度的原始轨迹流
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    # 输出层是 26 个神经元，表示26个英文字母的输出概率。
    keras.layers.Dense(26, activation='softmax')
])

# 编译模型：adam 优化器，稀疏交叉熵损失函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




# %% [markdown]
"""
我们要让数据在网络里跑大概几百轮 (epochs)，让模型持续自动提取特征和调优参数。

"""

# %%
print("开始训练神经网络...")
# batch_size 代表每次并行计算 1500 个样本； epochs=1000 代表整批数据将被刷 1000 层
# 我们把 verbose 设置为 0 （或者不频繁输出，保持整洁），只保留最终评估。
history = model.fit(X_train_dl, y_train_dl_num, batch_size=1500, epochs=1000, verbose=0)

print("模型训练完成，下面进行评估：")
test_loss, test_acc = model.evaluate(X_test_dl, y_test_dl_num, verbose=0)

print(f"深度学习测试集准确率: {test_acc*100:.2f}%")




# %% [markdown]
"""
可以看到，我们的模型**省去了手工发掘25个复杂结构变量的痛苦过程**，仅仅看了基础轮廓坐标点，就能取得极其接近乃至超过传统精密人工特征模型的识别准确度。这就是深度学习所谓的**端到端（End-to-End）自动特征提取**魔法！

## 3. 绘制混淆矩阵查错
我们可以利用 Sklearn 的功能或者 Matplotlib 绘制出 26 个字母的混淆矩阵，来看看神经网络最容易将哪些字母混淆。

"""

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 预测测试集
y_predict_prob = model.predict(X_test_dl)
y_predict_num = np.argmax(y_predict_prob, axis=1) # 选出概率最大的作为预测
y_predict_letters = [dl.number2Letter(num) for num in y_predict_num]

classes = dl.getAlphabet()
con_mat = confusion_matrix(y_test_dl, y_predict_letters, labels=classes, normalize="true")

plt.figure(figsize=(10, 8))
sns.heatmap(con_mat, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("深度学习模型 字母识别混淆矩阵")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()




# %% [markdown]
"""
## 4. 分析误分类情况 (Misclassifications)

在模型评估中，除了整体准确率外，观察**特定类别**的错误率往往能给我们更多启发，特别是深度学习这种“黑盒”模型。让我们统计一下哪些字母最容易被模型看错。

"""

# %%
# 提取误分类的数据：对角线是正确预测，非对角线是错误预测
yes = np.diag(con_mat)
no = con_mat - np.diag(yes)

# 计算每个字母的错误率
misratebyletter = np.sum(no, axis=1) / np.sum(con_mat, axis=1)

# 将错误率转为 DataFrame 并按错误率从高到低排序
misrate_by_letter = pd.DataFrame({
    'Letter': classes,
    'MisClassRate': misratebyletter
})
misrate_by_letter.sort_values(by="MisClassRate", inplace=True, ascending=False)

# 画出错误率条形图
plt.figure(figsize=(10, 4))
plt.bar(x=misrate_by_letter['Letter'], height=misrate_by_letter['MisClassRate'], color='salmon')
plt.title("深度学习模型 各字母误分类率 (Misclassification Rate)")
plt.ylabel("误分类率")
plt.xlabel("字母")
plt.show()




# %% [markdown]
"""
通过以上条形图和刚才的混淆矩阵，我们会发现有些字母（如 U 和 V，或者 M 和 N）在轨迹上确实极其接近。这就解释了为什么模型会犯特定的错误，这也是为什么后续我们还会想要通过更换更先进的网络架构（如 CNN 或 RNN）来改善这种识别精度。

"""

# %% [markdown]
"""
关于在这个手写字母识别任务中深度学习与传统机器学习的最大区别，以下哪项描述是最准确的？


**选项:**
- 深度学习必须依赖人工数学计算公式去提取每一个关键特征
- 深度学习属于端到端训练，会自动从简单数据中逐层提取关键特征特征
- 深度学习模型总是比所有机器学习方法轻量化且跑得更快

**正确答案:** 深度学习属于端到端训练，会自动从简单数据中逐层提取关键特征特征
**提示:** 端到端（End to End）最大的优势就是替人类专家省下了手动提取‘眼角位置’、‘笔画长度’等琐碎的 Feature Engineering 工作。

"""
