# %% [markdown]
"""
---
message: "本节课介绍信号处理三大核心技术：插值、重采样与快速傅里叶变换 (FFT)。"
---

# 信号处理基础：插值、重采样与 FFT

在工程应用中，我们经常需要应对因为传感器采样率不同导致的信号错位，或者电磁干扰带来的杂波。掌握了频域分析，你就相当于拥有了透视眼，能够一眼看穿隐藏在时间波形背后的本质规律。

**本节课你将学到：**

1.  **插值 (Interpolation)**: 当传感器采样率不够导致数据点过少时，如何用数学方法在已有的点之间“合理地”估算出新的数据点，从而恢复平滑的曲线。
2.  **降采样 (Downsampling)**: 当数据量过多，或是需要对齐不同时钟周期的设备数据时，如何快速削减数据点。
3.  **快速傅里叶变换 (FFT)**: 将包含各种频段以及环境噪声的复杂波形，从普通的“时域（振幅-时间）”映射到“频域（振幅-频率）”。就像是将一锅熬好的复合热汤还原成了一张清晰的配方表。
"""




# %% [markdown]
"""
## 1. 时间信号的插值 (Interpolation)

在现实世界中，物理量变化是连续的，但计算机记录下的数据是离散的。如果采样的点太少，画图就会变成像锯齿一样的折线，这有悖于自然界的连续规律。插值（Interpolation）就是通过多项式曲线拟合，在这几个稀疏的点之间补零、进而“捏造”出符合平滑规律的新点的过程。

假设有一组周期波动信号，我们因为设备算力不足只采集了寥寥 10 个数据点。直接把它们连成直线非常生硬。为了在稀疏点之间推算出合理的数据，我们常用以下数学方法的插值：

**物理与数学原理：**

1. **线性插值 (Linear Interpolation)**
   这是最简单的插值法。假设相邻两已知点为 $(x_0, y_0)$ 和 $(x_1, y_1)$，它们之间的任意一点 $x$ 对应的 $y$ 值通过几何比例计算：
   $$ y = y_0 + (x - x_0) \frac{y_1 - y_0}{x_1 - x_0} $$
   这种方法计算极快，但在连接处会有明显的“折角”，即一阶导数不连续，这在物理世界中通常违背了真实规律。

2. **三次样条插值 (Cubic Spline Interpolation)**
   为了重建符合物理惯性的平滑表现，工程师更喜欢样条插值。它的原理是在每两个相邻的数据点之间，构造一个三次多项式分段函数：
   $$ S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3 $$
   求解算法不仅要求曲线精确穿过每个已知点，还**强制要求在各点连接处的一阶导数（变化率）和二阶导数（曲率）都连续相同**。这就像是用一根富有弹性的柔性钢条（Spline）强行穿过这些点形成的自然弯曲形状。

调用 SciPy 的 `interp1d` 工具，我们可以直接生成对应策略的插值拟合函数。

"""

# %%
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 原始很稀疏且不足的数据 (只录入 10 个点)
x_raw = np.linspace(0, 2*np.pi, 10)
y_raw = np.sin(x_raw)

# 我们希望在同等时间宽度内，重构获得细密的 50 个时间刻度
x_dense = np.linspace(0, 2*np.pi, 50)

# 1. 线性插值 (Linear) - 会返回一个插值“函数对象”，随后输入新坐标即可
f_linear = interp1d(x_raw, y_raw, kind='linear')
y_linear = f_linear(x_dense)

# 2. 三次样条插值 (Cubic) - 对于波动信号这种能恢复出非常丝滑的物理极值点
f_cubic = interp1d(x_raw, y_raw, kind='cubic')
y_cubic = f_cubic(x_dense)

# 绘图比对插值重塑效果
plt.figure(figsize=(10, 5))
plt.plot(x_raw, y_raw, 'ko', markersize=8, label='Raw Data (10 points)')
plt.plot(x_dense, y_linear, 'r--', label='Linear Interpolation')
plt.plot(x_dense, y_cubic, 'b-', label='Cubic Interpolation')

plt.title("Interpolation Methods Comparison")
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
"""
## 2. 降采样与重采样 (Downsampling & Resampling)

重采样的逻辑与插值正好相反。比如设备 A 处理模块慢，每秒只能算 100 次，但它接收的电表源信号是原生 1000Hz 记录的。这就需要减少时间采样密度。当采样比率恰好是整数倍的时候（比如 1/10），最直接快速的做法就是 Python 原生的切片功能。

虽然 SciPy 有专门复杂的重采样函数，但如果只是简单的同比例提取抽样，直接通过 Python 的 `[::间隔]` 能够节省算力并立刻实现。

**非整数倍的重采样：**
如果我们需要把频率降低到非整数倍（例如把 50 个采样点降到 30 个点），简单的切片就不管用了。此时可以使用 `scipy.signal.resample`，它利用傅里叶变换原理，能够非常平滑优雅地在指定数量的节点之间重新分布出新的数据采集点。

"""

# %%
from scipy import signal

# 假设这是原始极高频的密集数据 (共 50 个点)
x_dense = np.linspace(0, 2*np.pi, 50, endpoint=False)
y_dense = np.sin(x_dense)

# 1. 整数倍降采样：按照每 5 个点抽取保留 1 个执行暴力切片降采样 (降到 10 个点)
step = 5
x_downsampled = x_dense[::step]
y_downsampled = y_dense[::step]

# 2. 非整数倍重采样：使用 signal.resample 降维至不多不少 30 个点
# 强调：它能同时帮你等比例重新划好新的对应时间/间隔横坐标，所以可以通过 t=x_dense 传入原横轴
y_resampled, x_resampled = signal.resample(y_dense, 30, t=x_dense)

plt.figure(figsize=(10, 5))
plt.plot(x_dense, y_dense, 'b-', alpha=0.4, label='Original (50 pts)')
plt.plot(x_downsampled, y_downsampled, 'go', markersize=8, label='Sliced Downsample (10 pts)')
plt.plot(x_resampled, y_resampled, 'r*', markersize=8, label='scipy.signal.resample (30 pts)')
plt.title("Downsampling (Integer) vs Resampling (Non-integer)")
plt.legend()
plt.grid(True)
plt.show()





# %% [markdown]
"""
## 3. 快速傅里叶变换 (FFT) 与 频域分析

这里有一个经典理论先提醒你，叫 **奈奎斯特(Nyquist)采样定理**：你设备上的采样频率 `fs`，起码必须是你期望测量的信号最高频域的 **2 倍**，计算机才能不失真地认出这个被测波形。这也是为什么示波器频率动辄上兆赫兹的原因。

终于到了整个信号处理的最重磅工具 —— FFT。为了模拟真实的恶劣工业环境，我们下面用代码生造一个“极度混乱”的信号：我们人为地把 300Hz、500Hz 和 700Hz 的独立完美波叠加在一起，甚至再丢进去一团巨大的高斯白噪声。你可以看到打印出来的原始时域图像杂乱无章，任何物理规律好像都已经消失了。

"""

# %%
from scipy.fft import fft, ifft

# --- 模拟带有大噪声的工业设备多谐波复合信号 ---
fs = 6000.0  # 我们定义系统的采样录入频率为 6000 Hz
time = 0.5   # 我们录取它短短 0.5s 的窗口片段
N = int(time * fs) # 根据公式计算出这段时间共有 3000 个采样数据点 (N)
t = np.linspace(0, time, N, endpoint=False) # 构造出对应 0 - 0.5 秒的主时间轴序列

# 将各个频率糅合叠加：300Hz(振幅10) + 500Hz(振幅20) + 700Hz(振幅30) + 随机白噪声
y = 10 * np.sin(2 * np.pi * 300 * t) + \
    20 * np.sin(2 * np.pi * 500 * t) + \
    30 * np.sin(2 * np.pi * 700 * t) + \
    np.random.normal(0, 5, N)

# 我们绘制最开始采集它时的时域波形（截取最前面的 100个采样点看局部细节）
plt.figure(figsize=(10, 4))
plt.plot(t[:100], y[:100])
plt.title("Time Domain Signal (Chaos with Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Measurement Value")
plt.grid(True)
plt.show()





# %% [markdown]
"""
### 3.1 FFT 频率谱与真实振幅【归一化】

现在见证奇迹：我们只需调用一行 `scipy.fft.fft(y)` 处理刚才的乱波段。该函数返回的内容全是包含实部虚部混合的**【复数数组】**！这意味着：
**对于这些复数代表的物理意义：其大小绝对值代表了这股波的振幅特征，而复数的角度则反映了波的初始相角。**

数学层面直接变换出的“双边谱”，是以折半为中心左右完全互相对称的图像，这部分完全没用可以抛弃。为了让计算出的数据值恢复成物理意义上最真实的频率谱图（即振幅还原回代码设置的10、20、30高度）：
1. 提取复数的绝对值成为日常的粗算振幅规模：`np.abs(fft_y)`。
2. 剥除无用的后半段“负面倒影”：把数组长度切片只保留到 `N // 2` 处（只留下有效正频率）。
3. 因被正反面对称分散的计算能量，单边的幅值需要【除以前面一半的样本长度 (`N / 2`)】缩放，才能完美恢复成设备原始振动大小的物理真值。

"""

# %%
# 1. 对上一个框中的原始杂乱信号执行 FFT
fft_y = fft(y)  
print(f"傅里叶处理返回数组长为: {len(fft_y)}，与源数据一模一样。")
print("可以发现前3个数值都是有实部虚部的复数: ", np.round(fft_y[:3], 1)) 

# 2. 我们通过运算分类提取其特征：算其绝对值(abs)可提原始振幅轮廓，算偏角(angle)得相位 
abs_y = np.abs(fft_y)
ang_y = np.angle(fft_y)

# 3. 构造与原信号一致的双向对称频率尺 f_bilateral 用于绘图说明：
f_bilateral = np.linspace(0, N-1, N) / N * fs  

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.title("Bilateral Amplitude Spectrum (Unnormalized, Mirror effect exists)")
plt.plot(f_bilateral, abs_y)  # 能明显看到对称的两侧峰林并立
plt.subplot(212)
plt.title("Bilateral Phase Spectrum")
plt.plot(f_bilateral, ang_y)
plt.tight_layout()
plt.show()

# -------------- 最重要的清理分水岭 --------------

# 4. 执行真正的复数绝对值清洗以及规整折算 (Normalization)
half_N = N // 2
half_f = f_bilateral[:half_N]            # 丢弃后边不要，横轴范围被限制到物理可见半轴 [0 ~ fs/2] 
normed_abs_y = abs_y[:half_N] / half_N   # 这一步极其重要：用其自身代表振幅特性的绝对值数组去除了规模常数。

plt.figure(figsize=(10, 4))
plt.plot(half_f, normed_abs_y) # 此时横轴已经是直接读取的真正频率（Hz），纵轴是真实振幅
plt.title("Unilateral Amplitude Spectrum (Clean & Normalized)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Real Calibrated Amplitude")
plt.xlim(0, 1000) # 由于上面最高也就 700Hz 目标，切出 0~1000 的部分近距离看细节
plt.grid(True)
plt.show()

# 极其清爽的频谱！那些躲藏在噪声下面看不见的 300, 500, 700 频率成分（分别准确逼近 10, 20, 30 的幅高度）毫无遮掩地裸露出来了。





# %% [markdown]
"""
### 3.2 强悍的频域带通滤波 (Bandpass Filter) 与 IFFT 反变化

频谱分析不仅仅是画图好看。既然已获得了这种魔法复数数组矩阵分布，我们可以就像拿着数字手术刀一样直接切除不想要频点段！

如果我们目标中只期望留下干净的 500Hz 本源信号段（大约位于 400 到 600Hz 之间）。只需要将被不需要的部分范围对应的数组粗暴【改写值为 0】，最后再来一发 `ifft` 就能反变换重刻出最完美干脆的原生时空图！
> ⚠️**防坑警告**：因为刚才操作的 `fft_y` 复数矩阵同时含有正反两个镜像段结构（见上图并立的双刺）。因此凡对前方频率置 0 的修改操作，必须去后方对称部分区域同步归 0 抹平，才能避免反推失真计算！

"""

# %%
low_idx = int(400 * N / fs)  # 计算 400Hz 占位在物理频长数组中的角标数索引
hi_idx  = int(600 * N / fs)  # 计算 600Hz 位置 

fft_y_filtered = fft_y.copy() # 万不可污染前面的原复数数组

# 抹除动作 A：清理低频项 (包含位于前部正侧 low_idx 前的所有数全清 0；且同时对称清理末尾对应倒数的同宽结构)
fft_y_filtered[:low_idx] = 0
fft_y_filtered[-low_idx:] = 0

# 抹除动作 B：清理过高频项 (这部分横跨中心点对称区，则把从中心往两头倒推 hi_idx 前的区域干掉)
fft_y_filtered[hi_idx:-hi_idx] = 0

# 调用万能逆变换函数 ifft，魔法还原
y_inversed = ifft(fft_y_filtered)

# 直接对比时域变化成果：
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.title("Original Time Waveform (Extremely Noisy & Unreadable)")
plt.plot(t[:100], y[:100])

plt.subplot(312)
plt.title("Recovered Target Signal (Thanks to Bandpass Filter, the 500Hz wave returns)")
plt.plot(t[:100], np.real(y_inversed)[:100], color='orange') # 逆变换出来还带微弱残留复数小数，为绘图用 np.real 提取其实部还原

plt.subplot(313)
plt.title("Ideal pure 500Hz reference comparison")
plt.plot(t[:100], (20 * np.sin(2 * np.pi * 500 * t))[:100], color='green')

plt.tight_layout()
plt.show()




# %% [markdown]
"""
### 3.3 进阶实战：直接锁定捕捉角度相位差

FFT 的最后一个实用级彩蛋是测相位：比如在检测变电站的传输压差或时间不同步时，我们发现同在一个基础频率（如500Hz）处，有两个波虽然同频但重合不上。这可以直接运用 Numpy 读取刚刚 FFT 同一对应频率点上复数值之中的【复数角度/相角】，二者一比较即可直接获得原始度数偏差。

我们手写两个标准的、初始存在滞后/超前相位差异的 500Hz 波（人为编入 $\pi/3$ 和 $\pi/6$的初始偏移相角），以供系统验算。

"""

# %%
# 这里我们顺用顶上设置好的基础频率尺
y_1 = np.sin(500 * 2 * np.pi * t + np.pi / 3) 
y_2 = np.sin(500 * 2 * np.pi * t + np.pi / 6)

# 我们让算法定格并锁在指定的 500Hz 的专属数组对应存放槽位处（idx）
idx_500 = int(500 * N / fs)

# 通过 np.angle 强行将这个复数的属性抽出来解读代表最初的始点角度
phase_delta = np.angle(fft(y_1)[idx_500]) - np.angle(fft(y_2)[idx_500])

print(f"通过 FFT 技术由复数值测出的推导相位差为: {phase_delta:.4f} (rad)")
print(f"代码真实掺入的强制相位差 (理论相减 pi/3 - pi/6): {np.pi/3 - np.pi/6:.4f} (rad)")





# %% [markdown]
"""
### 综合检测：解析隐藏波的“真实振大起底”

最后我们来一个清爽但考察很彻底的小检测。经过刚才的学习你应该明白：工业中执行原版的 `scipy.fft.fft` 最后拿到的复数数组根本没法直接给人眼直接识别看物理意义上的振强波幅。

**你的工程师任务：**
现有一段包含 $50\text{ Hz}$（原始物理波动高度/振幅被设定为了 $1.0$）和 $200\text{ Hz}$（异样振幅高强度达到了设为 $2.5$）的复合段。它已被转化好装在了 `fft_result` 复数数组中待提取。

怎么把它真面目揭露出来证实猜想呢？
1. **求绝对值**：面对复数不知道大小？请最简单地求取绝对值（有的也叫模长，也就是 `np.abs(处理复数值)` ）。
2. **规范化缩放并切除后段一半**：将得出的绝对值数组狠狠【除以有效量的一半（N/2）】规整信号量级还原本质刻度。然后并把不想要的反面后半段直接用切片舍弃不要了！
3. 请按照指引填空完成 TODO 任务！你的输出应当最终严格相等于 `1.0` 与 `2.5`。

"""

# %%
import numpy as np
from scipy.fft import fft

# 构建了一个工业测试系统：时长 1秒
fs = 1000.  
t = np.linspace(0, 1, int(fs), endpoint=False)
N = len(t) # 信号总采样点数量为 1000 个

# 截获一段电信号！它里面混带了双峰：50Hz（振动幅度 1.0）还有高频 200Hz（振动高度竟然拉到了 2.5）
y = 1.0 * np.cos(2 * np.pi * 50 * t) + 2.5 * np.sin(2 * np.pi * 200 * t)

# 执行完傅里叶变换，得到了满是虚虚实实的乱码复数数组 (`fft_result`)
fft_result = fft(y)

# TODO 1: 计算复数数组的绝对值（也就是信号的幅值特征）
abs_y = ...

# TODO 2: 对幅值进行归一化缩放，并切除掉因对称产生的无效的后半段镜像内容
# 提示：除以样本数量的一半来归一能量，然后利用切片保留前一半的数据
half_N = ... 
normed_amplitude = ...

# 下方已布置好自测，若操作无误应输出原信号构建值一致： 1.00 与 2.50
print(f"50Hz 主波段提纯出来的振幅高度验证: {normed_amplitude[50]:.2f}")
print(f"200Hz 强压干扰波提取的高度验证: {normed_amplitude[200]:.2f}")
