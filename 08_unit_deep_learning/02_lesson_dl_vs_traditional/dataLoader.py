#%% import
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

#%% 函数定义区域 (def)

def readData(fileDir):
    """
    遍历指定目录下的所有 csv 文件，读取并处理每一条手写轨迹。
    返回特征矩阵 (features) 和对应的标签列表 (labels)。
    """
    # 首先创建一个空的 np 数组用于存储特征，形状为 (1, 60)，稍后我们会由于逐行拼接而丢弃这冗余的第一行
    features = np.empty((1, 60))
    labels = []
    
    # 遍历文件夹中的所有文件
    for file in os.listdir(fileDir):
        # 拼接文件的完整路径
        filePath = os.path.join(fileDir, file)
        
        # 调用 getFeature 函数读取并处理单个文件，将结果展平为 (1, 60) 的一维向量
        feature = getFeature(filePath).reshape((-1, 60))
        
        # 将处理好的样本特征拼接（按行追加）到特征矩阵中
        features = np.concatenate((features, feature), axis=0)
        
        # 从文件名中提取标签，并存入标签列表中
        label = getLabel(file)
        labels.append(label)
        
    # 第一行是我们初始化时留下的空行，需要将其丢弃
    return (features[1:, :], labels)


def getFeature(filePath):
    """
    读取单一 csv 文件，进行数据清洗（去重、去空），并对坐标进行中心化与时间轴插值对齐处理。
    最终返回统一长度为 60 的一维特征向量（前30个是 X 坐标，后30个是 Y 坐标）。
    """
    # 使用 pandas 将 csv 文件读取为数据表
    data = pd.read_csv(filePath)
    
    # 删除重复的行。由于某些时间点记录可能有重复，如果不删除在后续做插值时会引发问题
    data.drop_duplicates(inplace=True)
    # 删除含有 NaN（缺失值）的行
    data.dropna(inplace=True)
    
    # 修正坐标的横纵宽高比 (aspect ratio) 以还原书写比例
    data.loc[:, "X"] = data.loc[:, "X"] * 1.5
    
    # 将整条轨迹做中心化处理（即让所有坐标减去它们的均值，使轨迹中心偏移到原点）
    data.loc[:, "X"] = data.loc[:, "X"] - data.loc[:, "X"].mean()
    data.loc[:, "Y"] = data.loc[:, "Y"] - data.loc[:, "Y"].mean()
    
    # 将 pandas 数据表转换为 numpy 数组以便后续矩阵计算
    dataArray = data.to_numpy()
    
    # 核心步骤：时间轴插值（Interpolation）
    # 目的是让每一条手写轨迹，不管它记录了多少个点、书写速度多快，都被强制在时间轴上均匀采样出固定数量的数据点（此处设为 30 个点），从而对齐特征长度。
    # 我们放弃了原始的 Time 列信息（视作所有字母用完全相同的时间写完），但由于采用等分时间差值，坐标点之间的跨度依然保留了书写者“速度的波动”。
    
    # 在头尾时间点之间，均匀生成 30 个时间点；并对原始的时间(dataArray[:,0])和 X、Y 坐标进行线性插值
    newX = np.interp(np.linspace(dataArray[0, 0], dataArray[-1, 0], 30),
                     dataArray[:, 0], data.loc[:, "X"])
    
    newY = np.interp(np.linspace(dataArray[0, 0], dataArray[-1, 0], 30),
                     dataArray[:, 0], data.loc[:, "Y"])
    
    # 将 30 个 X 坐标和 30 个 Y 坐标首尾相接，拼凑成长为 60 的一维特征向量 (2*30 -> 1*60)
    feature = np.concatenate((newX, newY), axis=None)
    return feature


def getLabel(fileName):
    """
    使用正则表达式，从标准格式的文件名中（例如 '../../data/08_unit_deep_learning/Data_csv/user001_A_1.csv'）提取出正确的英文字母标签（'A'）。
    """
    # 匹配规则：下划线前任意字符 _ 目标单字符(分组2) _ 任意字符.csv
    matchstr = '(.*?)_(.)_(.*?).csv'
    letter = re.match(matchstr, fileName).group(2)
    return letter


# ---------------- 辅助：标签编码转换 ----------------

def letter2Number(letter):
    """
    由于神经网络通常只能接受纯数字输入（作为目标变量），我们需要将大写字母转换为 0-25 之间的索引数字。
    例如：'A' -> 0, 'Z' -> 25
    """
    a2z = getAlphabet()
    return a2z.index(letter)             

def number2Letter(index):
    """
    模型预测输出的是 0-25 之间的数字，该函数将其反向映射回对应的大写字母。
    """
    a2z = getAlphabet()
    return a2z[index]

def getAlphabet():
    """
    生成并返回一个包含从 'A' 到 'Z' 共26个大写字母的有序列表。
    """
    alpha = 'A'
    test_list = []
    for i in range(0, 26): 
        test_list.append(alpha) 
        alpha = chr(ord(alpha) + 1)
    return test_list


