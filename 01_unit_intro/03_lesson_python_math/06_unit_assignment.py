# %% [markdown]
"""
# 单元综合练习：计算三维空间物体的质量

恭喜你学完了第一单元！作为本单元最终的实战测验，我们来解决一个三维空间的工程数学问题。

### 任务描述
在一个 3 维空间内，一种非均匀物质的密度分布为：
$$ \rho = x^2 y^2 z^2 $$

现有一个**圆锥体**，满足以下空间分布限制：
- 底面位于 $xy$ 平面，即 $z=0$
- 圆心在原点 $(0, 0, 0)$
- 半径为 $r = 1$
- 高度为 $h = 1$ （它的锥顶点在坐标 $(0, 0, 1)$ 位置）

在这个挑战中，你需要结合前面所学的知识，将物体形状转化为**三重积分**的上下限条件，并利用 `scipy.integrate.tplquad` 函数，求出该圆锥体的总质量 $M$。

*(提示：根据要求，我已经给你列出了 x, y, z 逐层积分的几何边界条件，按照对应的 `lambda` 语法填入函数即可。积分结果应当等于 $\frac{\pi}{6048}$)*

"""

# %%
import numpy as np
from scipy import integrate

# 要求：
# 1. 定义密度函数 (注意 tplquad 要求函数参数顺序为最内层到最外层：z, y, x)
def density(z, y, x):
    # 密度函数 rho = x^2 * y^2 * z^2
    return x**2 * y**2 * z**2
    
# 2. 调用 tplquad 进行三重积分求解
# x范围: -1 到 1
# y范围: 下界和上界依赖于x (即 -sqrt(1-x^2) 到 sqrt(1-x^2))
# z范围: 下界和上界依赖于x和y (即 0 到 1-sqrt(x^2+y^2))
# 圆锥侧面的方程可以写成 z = 1 - sqrt(x^2 + y^2)
# 所以对于每一个 (x, y)，z 都是从底面 0 积分到圆锥表面
total_mass, error = integrate.tplquad(
    density,
    -1, 1,
    lambda x: -np.sqrt(1 - x**2),
    lambda x: np.sqrt(1 - x**2),
    lambda x, y: 0,
    lambda x, y: 1 - np.sqrt(x**2 + y**2)
)

# 已知理论结果应为 pi / 6048，这里顺手做一个数值验算
expected_mass = np.pi / 6048

print(f"该圆锥体的总质量为: {total_mass:.6f}")
print(f"理论值为: {expected_mass:.6f}")
print(f"两者误差为: {abs(total_mass - expected_mass):.2e}")
