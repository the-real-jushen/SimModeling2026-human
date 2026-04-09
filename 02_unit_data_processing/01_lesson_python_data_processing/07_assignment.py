# %% [markdown]
"""
# 第二单元综合实战作业

经过前面的学习，你已经掌握了利用 Pandas 进行复杂数据整理清洗，以及利用 SciPy 进行高级信号变换与参数拟合的硬核技能。

本节为你准备了两个与真实世界接轨的综合挑战作业。请你利用本单元学到的所有知识独立完成。不要害怕遇到 bug，查阅官方文档、打印中间变量都是非常好的排错手段！

"""

# %% [markdown]
"""
## 作业 1：全球疫情数据追踪 (Pandas 实战)

**背景与任务：**
全球在经历了 COVID-19 疫情后积累了大量的公共卫生数据集。现在你需要对这些未经深度加工的原始表格进行梳理提取。

数据集中包含 `Date`（日期）、`Country/Region`（国家/地区）、`Confirmed`（累计确诊人数）等关键字段。

**工程师目标**：
1. **定位头目**：从数据集中精准提取并计算出，**历史上累计确诊总人数最多**的 5 个国家。
2. **趋势绘图**：分离出这 5 个国家的每日数据系列，利用 Matplotlib 绘制出这 5 个国家的**“确诊变化趋势折线图”**。
3. **增长分析**：计算并画出这 5 个国家的**“每日新增增长曲线”**（当日新增确诊人数）。

*在代码区已经为你写好了数据载入程序，请直接往下展开你的分析操作。*

"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# 已经为你准备好了数据读取代码
project_root = Path(__file__).resolve().parents[2]
covid_data_path = project_root / 'data/02_unit_data_processing/01_lesson_python_data_processing/covid-19-cases.csv'
cases = pd.read_csv(covid_data_path)

# 将宽表中的日期列提取出来，并按国家汇总各省/州数据
date_columns = [column for column in cases.columns if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2}', str(column))]
country_cases = cases.groupby('Country/Region')[date_columns].sum()

# 找出最终累计确诊人数最高的前 5 个国家
top_5_countries = country_cases.iloc[:, -1].nlargest(5)
print('累计确诊总人数最高的前 5 个国家：')
print(top_5_countries)

# 提取前 5 个国家的历史累计确诊序列，并将索引转为日期类型
top_5_history = country_cases.loc[top_5_countries.index].T
top_5_history.index = pd.to_datetime(top_5_history.index, format='%m/%d/%y')

# 绘制累计确诊趋势图
plt.figure(figsize=(12, 6))
for country in top_5_history.columns:
	plt.plot(top_5_history.index, top_5_history[country], linewidth=2, label=country)

plt.title('Top 5 Countries by Cumulative Confirmed Cases')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 使用差分计算每日新增病例；首日用累计值补齐，负值裁剪为 0
daily_new_cases = top_5_history.diff().fillna(top_5_history.iloc[0]).clip(lower=0)

# 绘制每日新增曲线
plt.figure(figsize=(12, 6))
for country in daily_new_cases.columns:
	plt.plot(daily_new_cases.index, daily_new_cases[country], linewidth=2, label=country)

plt.title('Daily New Confirmed Cases of Top 5 Countries')
plt.xlabel('Date')
plt.ylabel('Daily New Cases')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 作业 2：工业电网杂波分析与清洗 (SciPy 信号实战)

**背景与任务：**
某工厂上传了一份电表一秒钟的原始电压波动文件 `data.csv`。正常情况下电网的交流电应当是标准的纯粹基波（**幅值 220V，频率 50Hz**）。但这台设备极有可能遭受了别的机械共振或其他高频干扰（被称为谐波），导致原始数据极不规整。

我们的源文件传感器采样频率极高为 **10kHz**，持续截录了 **1s** 的数据时间。

**你的工程师任务：**
1. **降维重采样**：由于数据过大难以传递，请使用前面所学的切片法，将这 `10kHz` 的密密麻麻的数据压缩成 `1kHz` 分辨率。
2. **频域手术台**：对刚刚降频后的新数据实施 **FFT 快速傅里叶变换**。计算其频谱中的绝对值，揪出究竟混入了什么其他频率的捣乱谐波。*(提示：你可以在绘图中找峰值查看或打印出来，低微的干扰噪声不到 1% 可忽视)*
3. **精准带通滤波**：就像我们切除不要的数据一样，在获得的复数矩阵上，将目标基频（$50\text{Hz}$）以外的杂质频率全部强制赋零抹除。随后对其采取逆傅里叶变换 (`ifft`)。
4. **重建与对比**：绘制一个局部对比图，展示你通过剔除魔法所恢复出来的好波形，并与完美的公式标准波（$y = 220\sin(2\pi \cdot 50 \cdot t)$）画在一起比对。证明你的降噪重构非常成功！

*注：处理复数矩阵的赋值清零时，请千万留意镜像对称的后半段频段的清理！*

"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# 读取原始电网电压数据
voltage_data_path = '../../data/02_unit_data_processing/01_lesson_python_data_processing/data.csv'
df = pd.read_csv(voltage_data_path)
t_raw = df['Time'].to_numpy()
y_raw = df['Voltage'].to_numpy()

# 原始系统参数
fs_raw = 10000.0  # 数据源是极高频率的 10kHz
time_span = 1.0   # 截获时间总长 1s

# 提示：纯洁的电网主波应该是 50Hz 频率 (原始幅值为 220V)，但混入了多种干扰谐波！

# 暴力降采样：每 10 个点保留 1 个点，得到 1kHz 分辨率
downsample_step = int(fs_raw / 1000)
y_downsampled = y_raw[::downsample_step]
t_downsampled = np.arange(len(y_downsampled)) / 1000.0
fs_downsampled = 1000.0
n_samples = len(y_downsampled)

# FFT 频谱分析
spectrum = fft(y_downsampled)
freqs = np.fft.fftfreq(n_samples, d=1 / fs_downsampled)

# 构造单边频谱，并换算为更直观的物理幅值
positive_mask = freqs >= 0
positive_freqs = freqs[positive_mask]
positive_amplitude = 2 * np.abs(spectrum[positive_mask]) / n_samples
positive_amplitude[0] /= 2
if n_samples % 2 == 0:
	positive_amplitude[-1] /= 2

# 只打印占主导的谐波成分，忽略低于主峰 1% 的噪声
dominant_threshold = positive_amplitude.max() * 0.01
dominant_mask = positive_amplitude >= dominant_threshold

print('\n检测到的主要频率成分：')
for freq, amplitude in zip(positive_freqs[dominant_mask], positive_amplitude[dominant_mask]):
	print(f'{freq:.1f} Hz -> {amplitude:.2f} V')

# 频域滤波：仅保留正负 50Hz 的基频分量
filtered_spectrum = np.zeros_like(spectrum)
base_frequency = 50.0
frequency_tolerance = 1e-9
base_mask = np.isclose(np.abs(freqs), base_frequency, atol=frequency_tolerance)
filtered_spectrum[base_mask] = spectrum[base_mask]

# 逆变换回到时域，并取实部作为净化后的电压波形
y_clean = ifft(filtered_spectrum).real
y_ideal = 220 * np.sin(2 * np.pi * base_frequency * t_downsampled)

# 绘制频谱图，帮助观察主要谐波
plt.figure(figsize=(12, 5))
plt.stem(positive_freqs, positive_amplitude, basefmt=' ')
plt.xlim(0, 400)
plt.title('Single-Sided Amplitude Spectrum After Downsampling')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (V)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 绘制局部时域对比图
plot_mask = t_downsampled <= 0.08
plt.figure(figsize=(12, 5))
plt.plot(t_downsampled[plot_mask], y_clean[plot_mask], linewidth=2, label='Filtered Signal')
plt.plot(t_downsampled[plot_mask], y_ideal[plot_mask], '--', linewidth=2, label='Ideal 50 Hz Sine')
plt.title('Filtered Signal vs Ideal Power Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
