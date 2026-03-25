# %% [markdown]
"""
# 逻辑回归 (Logistic Regression)

前面我们学习的线性回归，都是用来预测**连续的数值**（如价格、长度）。但如果我们要预测的是**离散的类别**呢？比如：这封邮件是不是垃圾邮件？这个病人有没有得病？这个学生考试能不能及格？这就需要用到**逻辑回归 (Logistic Regression)**。

## 1. 为什么不能用线性回归做分类？

假设我们要预测学生根据学习时间是否能通过考试。
*   $X$：学习时间（小时）
*   $Y$：是否通过（0 表示挂科，1 表示通过）

如果我们强行用线性回归去拟合这些只有 0 和 1 的数据点，会得到一条直线。这条直线的问题在于：
1.  **预测值会超出范围**：直线的两端会延伸到小于 0 或大于 1 的区域，而概率不可能小于 0 或大于 1。
2.  **对异常值极其敏感**：如果有一个学神，学了 50 个小时才去考试（虽然他肯定能过），这个极端数据点会把整条直线拉偏，导致对普通学生的预测完全错误。

## 2. 逻辑回归与 Sigmoid 函数

为了解决这个问题，逻辑回归引入了一个神奇的函数：**Sigmoid 函数（也叫逻辑函数）**。

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

这个函数的特点是：
*   无论输入的 $z$ 有多大或多小，输出的值永远被“压缩”在 **0 到 1 之间**。
*   它的形状像一个“S”型曲线。

![Logistic Curve](https://upload.wikimedia.org/wikipedia/commons/c/cb/Exam_pass_logistic_curve.svg)
*(图示：逻辑回归曲线。横轴是学习时间，纵轴是通过考试的概率。数据点只有 0 和 1，但拟合出的 S 型曲线给出了在 0 到 1 之间的平滑概率预测。)*

**逻辑回归的模型：**
逻辑回归实际上就是把线性回归的结果 $z = \beta_0 + \beta_1 x_1 + \dots$ 塞进 Sigmoid 函数里：
$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots)}} $$

这样，模型的输出就变成了一个**概率值**（例如 0.85，表示有 85% 的概率通过考试）。如果我们设定一个阈值（通常是 0.5），大于 0.5 就预测为 1（通过），小于 0.5 就预测为 0（挂科），这就完成了一个**二分类 (Binary Classification)** 任务。

"""

# %% [markdown]
"""
让我们用代码来实现这个“学习时间与考试通过率”的例子。我们将使用 `scikit-learn` 库中的 `LogisticRegression` 模型。

## 3. 代码实战：预测考试通过率

下面是我们收集到的 20 名学生的学习时间与考试结果。**注意：在 1.75 小时处，有一个人挂了，一个人过了，这体现了现实数据中的随机性。**

| 学习时间 (小时) | 是否通过 (0=挂科, 1=通过) |
| :---: | :---: |
| 0.5 | 0 |
| 0.75 | 0 |
| 1.0 | 0 |
| 1.25 | 0 |
| 1.5 | 0 |
| 1.75 | 0 |
| 1.75 | 1 |
| 2.0 | 0 |
| 2.25 | 1 |
| 2.5 | 0 |
| 2.75 | 1 |
| 3.0 | 0 |
| 3.25 | 1 |
| 3.5 | 0 |
| 4.0 | 1 |
| 4.25 | 1 |
| 4.5 | 1 |
| 4.75 | 1 |
| 5.0 | 1 |
| 5.5 | 1 |

"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. 准备数据
# 学习时间 (小时)
hours = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5, 
                  2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5])
passed = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# reshape 为 sklearn 需要的二维数组格式
X = hours.reshape(-1, 1)
y = passed

# 2. 创建并训练逻辑回归模型
model = LogisticRegression()
model.fit(X, y)

# 3. 进行预测
# 生成一些用于画平滑曲线的测试点 (从 0 到 6 小时，生成 100 个点)
X_test = np.linspace(0, 6, 100).reshape(-1, 1)

# predict() 方法直接输出分类结果 (0 或 1)
y_pred_class = model.predict(X_test)

# predict_proba() 方法输出属于各个类别的概率
# 它返回一个二维数组，第一列是属于类别 0 的概率，第二列是属于类别 1 的概率
y_pred_prob = model.predict_proba(X_test)
# 我们通常只关心属于类别 1 (通过) 的概率
prob_pass = y_pred_prob[:, 1]

# 4. 可视化结果
plt.figure(figsize=(8, 5))

# 画出原始数据点
plt.scatter(X, y, color='blue', label='Actual Data (0=Fail, 1=Pass)', zorder=5)

# 画出逻辑回归拟合的 S 型概率曲线
plt.plot(X_test, prob_pass, color='red', linewidth=2, label='Probability of Passing')

# 画一条 y=0.5 的阈值线
plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')

plt.title('Logistic Regression: Study Hours vs Exam Pass Probability')
plt.xlabel('Study Hours')
plt.ylabel('Probability / Class')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()




# %% [markdown]
"""
### 结果分析原理

在上面的代码中：
1. **`predict()` 方法**：直接输出分类结果（0 或 1）。模型内部会默认以 $0.5$ 为分类阈值。
2. **`predict_proba()` 方法**：返回一个属于各个类别的二维概率数组，第一列为类别 0 的概率，第二列为类别 1 的概率。

**观察图表结果：**
1. **概率约束**：红色的 S 型曲线完美地将通过概率限制在 0 和 1 之间，解决了线性回归可能预测出超过此范围的问题。
2. **决策边界**：红线与灰线相交处（约为 2.7 小时）可以视为边界，当学习时间超过该界限，通过的概率就大于 0.5。
3. **渐近线特征**：随学习时间变长，通过考试的预测概率逐渐接近 100%，这高度符合我们对现实事物的认知。

---
message: "多元逻辑回归实战，演示如何使用多个特征（如年龄和收入）进行购买行为预测。"
---

## 4. 多元逻辑回归：预测客户购买行为

逻辑回归不仅可以处理一个特征，也可以处理多个特征。接下来，我们看一个更贴近商业应用的例子：根据客户的年龄和收入，预测他们是否会购买某件商品。

假设我们有以下 12 名客户的年龄 (Age)、收入 (Income) 数据，以及他们最终是否购买了商品 (Purchased: 0=否, 1=是)。

| 年龄 (Age) | 收入 (Income) | 是否购买 (Purchased) |
| :-------: | :----------: | :----------------: |
| 56 | 40627 | 0 |
| 69 | 38792 | 0 |
| 46 | 103969 | 0 |
| 32 | 73001 | 0 |
| 60 | 106552 | 0 |
| 25 | 53897 | 1 |
| 38 | 98148 | 1 |
| 45 | 80000 | 1 |
| 22 | 45000 | 1 |
| 35 | 60000 | 1 |
| 50 | 120000 | 0 |
| 28 | 55000 | 1 |

我们将通过建立多元逻辑回归模型，来预测新客户：**客户 A (25 岁, 收入 50000)** 和 **客户 B (45 岁, 收入 80000)** 购买该商品的概率。
"""




# %%
# 1. 准备数据
data = {
    'Age': [56, 69, 46, 32, 60, 25, 38, 45, 22, 35, 50, 28],
    'Income': [40627, 38792, 103969, 73001, 106552, 53897, 98148, 80000, 45000, 60000, 120000, 55000],
    'Purchased': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X_multi = df[['Age', 'Income']]
y_multi = df['Purchased']

# 2. 训练模型
model_multi = LogisticRegression()
model_multi.fit(X_multi, y_multi)

# 3. 预测新客户
# 客户 A: 25岁，收入 50000
# 客户 B: 45岁，收入 80000
new_customers = pd.DataFrame({
    'Age': [25, 45],
    'Income': [50000, 80000]
})

# 预测概率
probs = model_multi.predict_proba(new_customers)
# 预测类别
classes = model_multi.predict(new_customers)

print("--- 客户购买预测结果 ---")
for i, (index, row) in enumerate(new_customers.iterrows()):
    print(f"客户 {i+1} (年龄: {row['Age']}, 收入: {row['Income']}):")
    print(f"  购买概率: {probs[i][1]:.2%}")
    print(f"  预测结果: {'购买 (1)' if classes[i] == 1 else '不购买 (0)'}\n")




# %% [markdown]
"""
### 多元的逻辑结果分析

在多元逻辑回归中，输入 $z$ 变成了多个特征的线性组合：
$$ z = \beta_0 + \beta_1 \times \text{Age} + \beta_2 \times \text{Income} $$

通过模型预测可以看出：
- **客户 1（25岁，收入50000）**：在现有数据中，低龄群体大多倾向于购买（购买主力军），模型给出其购买概率较高。
- **客户 2（45岁，收入80000）**：尽管收入看似较高，但在已有数据分布下，高龄群体倾向于不购买，此预测反映了多维度的综合决策结果。

通过联合考察多个维度（年龄、收入），逻辑回归不仅能快速给出“是否购买”的类别分类，还能输出决策置信度（即概率数值），帮助企业实现更为精细的营销规划。

"""

# %% [markdown]
"""
## 5. 动手挑战：建立你的防流失预测模型

为了巩固前面的知识，请尝试独立完成一个**预测用户是否流失（Churn）**的逻辑回归模型。对于产品运营来说，判断用户是“继续深造”还是“即将弃用流失”，对制定挽留策略至关重要。

**小测验**：利用给定的用户活跃数据（APP使用时长与点击次数），完成逻辑回归模型的实例化、训练以及对新用户的预测。

"""

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 给定的用户活跃历史数据
train_data = {
    'App_Time': [120, 10, 55, 30, 80, 5, 20, 100],
    'Clicks':   [300, 20, 150, 40, 220, 5, 30, 250],
    'Churn':    [0,   1,  0,  1,  0,  1,  1,  0]  # 1=流失弃用，0=留存
}
df_train = pd.DataFrame(train_data)
X_train = df_train[['App_Time', 'Clicks']]
y_train = df_train['Churn']

# 测试需要预测的新客户数据 (10小时使用时长, 50次点击)
X_test = pd.DataFrame({'App_Time': [10], 'Clicks': [50]})

# --- 请在下方完善代码 ---
# 1. 实例化 LogisticRegression 模型
model = 

# 2. 用数据拟合训练模型


# 3. 预测类别结果 (输出结果赋值给 y_pred 变量)


# 4. 预测概率 (输出结果赋值给 y_prob 变量)


print("预测流失分类 (1=流失):", y_pred)
print("预测概率输出:", y_prob)
