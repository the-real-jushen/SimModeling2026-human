# %% [markdown]
"""
# 第八单元：深度学习基础 (Deep Learning Basics)
## 第一课：深度学习与人工神经网络简介

欢迎来到本课程的最后一个单元！
在前面的课程中，我们学习了传统的机器学习算法（如线性回归、逻辑回归、SVM、决策树等）。
今天，我们将初步探索目前人工智能领域最热门、最核心的技术：**深度学习 (Deep Learning, DL)**。

### 1. 什么是深度学习？它与传统机器学习的区别是什么？

深度学习是机器学习的一个子集（机器学习又是人工智能的一个子集）。
深度学习主要基于**人工神经网络 (Artificial Neural Networks, ANNs)** 的架构。

**最大的区别在于：特征提取 (Feature Extraction) 的方式**

![深度学习 vs 传统机器学习](images/dl_vs_ml.png)

1.  **传统机器学习 (Traditional ML):** 
    通常需要人类专家花费大量的时间和精力去**手动提取特征 (Manual Feature Engineering)**。
    例如，在识别手写数字时，人类需要告诉模型：“去提取笔画的长度、交叉点的数量、圆圈的数量”作为特征，然后再把这些特征输入给 SVM 或 KNN 模型。
    *缺点：* 面对复杂数据（如高清图片、视频、音频、自然语言），人工提取特征极其困难甚至是不可能的。

2.  **深度学习 (Deep Learning):**
    被称为**端到端学习 (End-to-End Learning)**。你只需要把原始数据（例如猫的图片的几十万个像素值）直接扔给神经网络，网络通过内部复杂的计算，**自动学习并提取**出从低级（边缘、轮廓）到高级（猫耳、猫眼）的特征，最后直接输出分类结果。
    *优点：* 省去了繁琐的人工特征工程。
    *缺点：* 极其依赖**大量的数据**和**强大的计算能力 (GPU)**，且模型内部的运作机制往往是一个难以解释的“黑盒 (Black Box)”。

### 2. 人工神经网络 (Artificial Neural Network) 的基本结构

神经网络的设计灵感来源于人类大脑的神经元结构。

![神经网络基本结构](images/neural_network_structure.png)

一个标准的只向前传递数据（不回头计算）的网络叫做 **前馈神经网络 (Feedforward Neural Network)**，它通常由三部分组成：

1.  **输入层 (Input Layer):** 负责接收外部数据。网络有几个输入特征，输入层就有几个神经元。
2.  **隐藏层 (Hidden Layer):** 也就是网络“深度”的由来。位于输入层和输出层之间，可以是一层，也可以是非常多层（深度网络）。这里的神经元负责进行复杂的数学变换（特征提取）。
3.  **输出层 (Output Layer):** 负责输出最终的预测结果（例如分类的概率或回归的连续数值）。

**神经元内部是如何工作的？**
每一个绿色的圈圈（神经元/节点）实际上做的是一个简单的数学运算：
$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
*   $x_i$: 从上一层传来的输入值。
*   $w_i$: 权重 (Weights)，表示这个连接的重要性。网络**“学习”**的过程，本质上就是不断调整这些权重的过程。
*   $b$: 偏置 (Bias)，用来平移计算结果。

算完了 $z$ 之后，还需要套上一个**激活函数 (Activation Function)** $a = f(z)$，将其变成非线性的。如果没有激活函数，无论神经网络有多少层，它最终都只是一个巨大的线性回归模型。

---

### 3. 使用 TensorFlow/Keras 构建你的第一个神经网络

为了演示深度学习，我们将使用 Python 中最流行的深度学习框架：**TensorFlow**（由 Google 开发）及其高级 API **Keras**。

我们将使用极其经典的 **MNIST 手写数字数据集**。这个数据集包含了 70,000 张 28x28 像素的灰度图片，内容是 0 到 9 的手写数字。我们的目标是训练一个神经网络，让它能认出图片里的数字是什么。

"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入 TensorFlow 和 Keras
import tensorflow as tf
from tensorflow import keras

# 检查 TensorFlow 版本
print(f"TensorFlow 版本: {tf.__version__}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载和预处理 MNIST 数据集
# Keras 直接内置了这个数据集，由于在国外服务器，初次下载可能较慢
mnist = keras.datasets.mnist

# 将数据分为训练集和测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"训练集图片形状: {X_train.shape}")  # (60000, 28, 28) 表示6万张28x28的图
print(f"训练集标签形状: {y_train.shape}")
print(f"测试集图片形状: {X_test.shape}")




# %% [markdown]
"""
让我们先看几张训练集里的手写数字图片到底长什么样。

"""

# %%
# 可视化前 5 张图片
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    # 取出第 i 张图（28x28 的矩阵），用灰度色彩映射显示
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"标签: {y_train[i]}")
    plt.axis('off') # 不显示坐标轴
plt.show()




# %% [markdown]
"""
图片的每一个像素值是一个 0 到 255 之间的整数（0 代表全黑，255 代表全白）。
在训练神经网络之前，我们要进行**归一化 (Normalization)** 操作，将像素值缩放到 0 到 1 之间，这会让神经网络学得更快也更稳定。

"""

# %%
# 归一化：将像素值从 0-255 缩放到 0.0-1.0
X_train = X_train / 255.0
X_test = X_test / 255.0

# 2. 构建神经网络模型
# 我们使用 keras 的 Sequential 模型（序列模型），也就是一层一层堆叠起来的网络
model = keras.Sequential([
    # 输入层：把 28x28 的二维图片矩阵“展平”成一维的 784 个像素点向量
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # 隐藏层：全连接层 (Dense)，包含 128 个神经元
    # 激活函数使用 'relu'（修正线性单元），它是目前最常用的隐藏层激活函数
    keras.layers.Dense(128, activation='relu'),
    
    # 还可以再加一个隐藏层（增加网络深度可以学习更复杂的特征，但也容易过拟合）
    # keras.layers.Dense(64, activation='relu'),
    
    # 随机丢弃层 (Dropout)：训练时随机断开 20% 的神经元，用于防止模型过拟合死记硬背
    keras.layers.Dropout(0.2),
    
    # 输出层：全连接层，包含 10 个神经元（因为我们的分类目标是数字 0 到 9，共 10 类）
    # 激活函数使用 'softmax'，它能输出 10 个类别的概率分布（它们的概率和为 1）
    keras.layers.Dense(10, activation='softmax')
])

# 查看我们刚刚搭好的网络结构摘要
model.summary()




# %% [markdown]
"""
### 4. 编译和训练模型

搭建好网络结构后，我们需要**编译 (Compile)** 模型，告诉它三件事：
1.  **优化器 (Optimizer):** 模型用什么方法来更新参数权重的。最流行的是 `adam`，比传统梯度下降要快得多。
2.  **损失函数 (Loss Function):** 用来衡量模型预测的错误程度。对于多分类问题，通常使用交叉熵 (`sparse_categorical_crossentropy`)。
3.  **评估指标 (Metrics):** 训练过程中我们不仅想看损失下降没，还想直观地看到**准确率 (accuracy)** 提升没。

"""

# %%
# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 训练模型
# 把训练数据 X_train 和标签 y_train 送进去
# epochs表示遍历整个数据集的次数
print("开始训练模型...")
# history 会记录下每一轮(epoch)的 loss 和 accuracy 信息
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
print("模型训练完成！")




# %% [markdown]
"""
观察训练打印的信息：
你可以在短短 5 轮迭代 (epochs) 中看到，模型在训练集上的准确度飞速上升到了 97% 以上！这就是神经网络的强大之处。
同时，`val_accuracy` 显示了模型在测试集（未见过的数据）上的表现，同样非常优秀，说明模型没有严重过拟合。

我们可以把训练过程中的准确率和损失画成折线图。

"""

# %%
# 绘制训练过程的 准确率变化 和 损失变化
plt.figure(figsize=(12, 4))

# 准确率图
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练集准确率 (Train Accuracy)')
plt.plot(history.history['val_accuracy'], label='测试集准确率 (Val Accuracy)')
plt.title('模型准确率变化')
plt.xlabel('Epoch (迭代轮数)')
plt.ylabel('Accuracy (准确率)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# 损失图
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练集损失 (Train Loss)')
plt.plot(history.history['val_loss'], label='测试集损失 (Val Loss)')
plt.title('模型损失值变化 (Loss越低越好)')
plt.xlabel('Epoch (迭代轮数)')
plt.ylabel('Loss (损失值)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.show()




# %% [markdown]
"""
### 5. 评估与预测

模型训练好后，让我们用测试集来最终评估一下它的准确性，并随便挑几张图片让它做一下预测。

"""

# %%
# 在测试集上进行整体评估
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"\n--- 测试集最终准确率: {test_acc*100:.2f}% ---")

# 使用模型对前 5 张测试图进行预测
# model.predict 会输出一个包含 10 个概率值的数组
predictions = model.predict(X_test[:5])

# 取出概率最大的那个索引，就是模型的最终预测结论
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test[:5]

# 可视化预测结果
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    
    # 如果预测对了用绿色标题，预测错了用红色标题
    color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
    plt.title(f"预测: {predicted_classes[i]}\n真实: {true_classes[i]}", color=color)
    plt.axis('off')
plt.show()




# %% [markdown]
"""
### 总结与展望

这就完成了！你已经成功使用 Keras 构建、训练并评估了你的第一个深度神经网络！
在这个简单的例子中，网络通过它拥有的数十万个连接权重，自动学会了区分各种杂乱笔画构成的手写数字，取得了惊人的准确率。

**后续你可以去探索的深度学习领域：**
1.  **卷积神经网络 (CNN):** 专门用来处理图片，是目前图像识别、自动驾驶视觉领域的核心技术（今天的例子其实更应该用 CNN 效果会更好）。
2.  **循环神经网络 (RNN) / Transformer:** 专门用来处理序列数据，如自然语言处理、机器翻译、语音识别。
3.  **强化学习 (Reinforcement Learning):** 让智能体不断试错来学习策略，这就是打败人类围棋冠军的 AlphaGo 背后的技术核心。

### 课堂练习
1.  **改变网络结构：** 回到上面构建模型的地方，修改代码，加宽隐藏层（例如把 128 改成 256）或者增加一个隐藏层，然后重新运行所有代码块。观察模型的训练速度、以及准确率是否有所提升。
2.  **增加训练时间：** 把 `model.fit()` 中的 `epochs=5` 修改为 `epochs=15`。观察准确率曲线的变化。到了后期，模型可能会出现怎样的现象？

"""
