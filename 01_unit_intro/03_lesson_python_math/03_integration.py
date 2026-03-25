# %% [markdown]
"""
# 3. 积分 (Integration): 空间的累加器

在物理和工程中，我们经常需要计算空间或时间的累加总量。
- 速度关于时间的积分是什么？是**位移**。
- 功率关于时间的积分是什么？是**能耗(电能)**。
- 截面积关于高度的积分是什么？是**体积**。

这就是积分。如果说微分是“拆解”（求局部变化率），那么积分就是“组装”（求总和）。
在 Python 的底层科研组件中，主要由 SciPy 库的 `scipy.integrate` 模块承担这一重任。

![Integration Area under Curve](https://avhol.com/wp-content/uploads/2023/07/Integration-Area-Under-the-Curve-1024x614.png)
*(无论是黎曼和、梯形法则还是辛普森法则，数值积分的核心思想都是：切分->测算面积->累加)*

"""

# %% [markdown]
"""
## 数值积分：SciPy 的 `quad` 函数

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/SCIPY_2.svg/300px-SCIPY_2.svg.png)

在实际建模中，我们极少使用符号积分来求解析解（公式），因为真实世界带摩擦力和空气阻力的方程往往**根本不存在完美的原函数**。
SciPy 提供了 `quad` 函数来进行**数值积分**（Quadrature，求积法）。它使用了高度优化的 FORTRAN 库（QUADPACK），能够在保证极高精度的同时飞速完成计算。

**目标：** 计算定积分 $\int_0^1 (ax^2 + b) dx$，其中 $a=2, b=1$。

"""

# %%
from scipy import integrate

# 第一种写法：标准函数定义
# 注意：作为被积函数，它的第一个参数必须是变量 x
def my_func(x, a, b):
    return a * (x**2) + b

# integrate.quad(被积函数, 下限, 上限, args=(函数额外的参数, ...))
# quad 会返回两个值：积分结果 和 误差估计值
result, error = integrate.quad(my_func, 0, 1, args=(2, 1))

print(f"[标准写法] 积分结果: {result:.6f}")
print(f"[标准写法] 绝度误差估计: {error}")

# ----------------------------------------------------
# 第二种写法 (推荐)：使用 Lambda 匿名函数
# 对于简短的代码，我们可以用 lambda 在一行内“临时”生成一个函数

result_lambda, error_lambda = integrate.quad(lambda x: 2*(x**2) + 1, 0, 1)

print(f"\n[Lambda写法] 积分结果: {result_lambda:.6f}")





# %% [markdown]
"""
### 动手写积分代码：广义积分计算

有时候积分的上限或下限是**无穷大**。
比如在计算一个随着距离平方衰减的磁场能量分布时，我们可能要计算：

$$ E = \int_1^{\infty} \frac{1}{x^2} dx $$

*(数学上通过极限易知，这其实收敛于数值 1)*。

在 NumPy 中，正无穷大可以用 `np.inf` 表示。
请你在下面的代码块中，运用 `integrate.quad` 和 `lambda` 函数，计算这个无穷区间的广义积分。

"""

# %%
import numpy as np
from scipy import integrate

# 请填空完成这个无穷积分：
# 使用 lambda x: 1 / (x**2) 作为被积函数
# 下限是 1，上限使用 np.inf
result, error = # TODO: 请在此处调用 integrate.quad

print(f"广义积分计算结果为: {result}")

# %% [markdown]
"""
## 终极形态：多重积分 (三重积分)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Triple_Integral_Volume.png/320px-Triple_Integral_Volume.png)

如果你在做热力学仿真，可能需要计算一个 3D 铁块腔体内部的总热量，这就需要用到三重积分 `tplquad` (Triple Quadrature)。

计算一个奇怪的三重积分：
$$ \int_0^1 \int_0^1 \int_0^1 (x^y - z) \,dz \,dy \,dx $$

**警告**：SciPy 的多重积分函数的参数顺序非常“反人类”。
`tplquad(func(z, y, x), x下/上, y下/上, z下/上)`
- 积分函数 `func(z, y, x)`的输入顺序是从内到外的：最内层积分变量排前面，最外层排后面。
- 但是后面的边界设定，边界却需要被定义成函数（即使它是常数），比如 `lambda x: 0`。

"""

# %%
from scipy import integrate

# 定义被积函数 (注意顺序：最内层 z, 中间 y, 外层 x)
def f_3d(z, y, x):
    return x**y - z

# 调用 tplquad 进行三重数值积分
I, error = integrate.tplquad(
    f_3d,
    0, 1,                          # 最外层 x 的常数边界: a, b
    lambda x: 0, lambda x: 1,      # 中间层 y 的边界(作为外层 x 的函数): gfun, hfun
    lambda x, y: 0, lambda x, y: 1 # 最内层 z 的边界(作为 x,y 的函数): qfun, rfun
)

print(f"三重数值积分求得的结果是: {I:.6f}， 误差极小为 {error:.2e}")
# 你会发现，这几行代码，秒杀了我们在上节课写的巨型 numpy for 循环黎曼和嵌套！




# %% [markdown]
"""
### 随堂测试

为了让代码更紧凑，如果在 Python 中我想要用一行代码定义一个“返回输入数值平方加上两倍输入数值”的匿名函数（用于传给积分函数），我应该使用哪种语法？


**选项:**
- `def f(x): return x**2 + 2*x`
- `lambda x: x**2 + 2*x`
- `function(x) { return x**2 + 2*x }`
- `x => x**2 + 2*x`

**正确答案:** `lambda x: x**2 + 2*x`
**提示:** 在 Python 中，匿名函数使用 `lambda` 关键字，后跟参数列表，冒号，然后是返回值表达式。

"""
