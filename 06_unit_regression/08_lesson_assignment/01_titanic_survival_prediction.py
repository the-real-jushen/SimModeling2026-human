# %% [markdown]
"""
# 综合测验：泰坦尼克号生存预测

大名鼎鼎的泰坦尼克号沉没事件是历史上最著名的海难之一。在这场灾难中，有很多人不幸遇难，也有生还者。人们发现，生存的概率与乘客的性别、年龄、舱位等级等因素存在着密切的关系。

## 任务目标

本测验要求你应用**逻辑回归 (Logistic Regression)** 模型，通过对过去乘客信息的训练，去预测其他乘客在这次事故中是否生还。这需要你进行基本的数据预处理（比如处理缺失值、将分类文本转换为数值等），并训练出逻辑回归模型。

## 1. 探索数据

我们将以一份名为 `titanic_data.csv` 的数据集开始。首先我们看一下这份数据长什么样子。

"""

# %%
import os
import pandas as pd

# 获取当前脚本所在目录 上升两级找到工作区目标，定位到 data 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
data_path = os.path.join(current_dir, '..', '..', '..', 'data', '06_unit_regression', '../../data/06_unit_regression/titanic_data.csv')

# 兼容相对路径回退方案
if not os.path.exists(data_path):
    data_path = '../../data/06_unit_regression/titanic_data.csv'

df_train = pd.read_csv(data_path)
print("数据集的前 5 行：")
print(df_train.head())




# %% [markdown]
"""
观察上面的数据，里面包含了很多字段：
- `Pclass`: 舱位等级 (1 = 头等舱, 2 = 二等舱, 3 = 三等舱)
- `Sex`: 性别
- `Age`: 年龄
- `SibSp`: 船上的兄弟姐妹/配偶数量
- `Parch`: 船上的父母/子女数量
- `Fare`: 票价
- `Embarked`: 登船港口
- `Survived`: 是否生还 (0 = 否, 1 = 是) - **这是我们将要预测的目标！**

## 2. 数据预处理
机器只能计算数字，所以我们需要把文本类别（比如性别、登船港口）转换成数字。在 Pandas 中，我们经常用 `pd.get_dummies()` 来实现**独热编码 (One-Hot Encoding)**，它能把文本转化为 0 和 1 的哑变量矩阵。

"""

# %%
def pre_processing_titanic(df):
    # 只选择我们需要的特征列以及目标列
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
    
    # 丢弃包含缺失值的行
    df = df.dropna()
    
    # 对具有文本类别的列（性别、登船港口）进行独热编码
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    
    # 分离特征集 X 和目标结果 Y
    X = df.loc[:, df.columns != 'Survived'].values
    y = df.loc[:, df.columns == 'Survived'].values
    
    return X, y

# 执行预处理，获得我们可以直接输入到模型中的数字矩阵 X 和 Y
X, y = pre_processing_titanic(df_train)
print(f"预处理后，特征矩阵 X 的形状: {X.shape}, 目标列 y 的形状: {y.shape}")




# %% [markdown]
"""
## 3. 进行训练与评估，看你的了！

接下来，你需要结合在第 6 单元所学习的内容，对数据进行训练集与测试集的切分，实例化一个逻辑回归模型拟合它，并最后绘制打印出评估结果（混淆矩阵）。

"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 使用新版 sklearn 里推荐的 ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. 拆分数据集，80%用于训练，20%用于测试
# X, y 已经在前面准备好
train_X, test_X, train_y, test_y = train_test_split(X, y.ravel(), test_size=0.2, random_state=42)

# 2. 实例化并训练 Logistic Regression 模型
# 请在下方补充代码完成实例化和拟合
log_model = 

# 3. 对测试集 test_X 进行预测
pred_y = 

# 4. 绘制混淆矩阵 (Confusion Matrix)
# 查看模型在测试集上预测对和预测错的数量
disp = ConfusionMatrixDisplay.from_estimator(log_model, test_X, test_y, cmap=plt.cm.Blues)
plt.title('Titanic Survival Prediction - Confusion Matrix')
plt.show()
