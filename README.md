# 电气工程建模与仿真（2026）

本项目为面向电气工程及相关专业的教学课程代码与材料集合，旨在通过 Python 实践掌握数学建模、数值计算、数据处理、优化与机器学习基础，并结合工程案例提升建模与仿真能力。

## 项目概览
- **课程结构**：分为 8 个单元（Unit 1–8），内容从建模概念与 Python 基础，到数据处理、动力学建模、优化算法、回归与分类、再到深度学习入门。
- **每单元内容**：每个单元含若干课（lesson），每课由若干 Python 脚本作为练习/示例（详见 `course.yaml`）。
- **目标读者**：电气工程或相关专业的本科/研究生学生，以及希望通过工程案例学习建模与仿真流程的工程师。

## 主要文件与目录
- `course.yaml`：课程元数据与结构描述（章节、课次与示例脚本）。
- `01_unit_intro/` … `08_unit_deep_learning/`：按单元组织的教学代码与示例。每个课目录下包含若干 `.py` 脚本与图片资源。
- `data/`：用于练习或示例的数据文件（部分文件较大，见下面注意事项）。


## 环境配置与依赖安装（2026年3月最新说明，推荐流程）

本仓库已适配中国大陆常见网络环境，推荐如下步骤极速完成环境搭建：

1. **创建并激活虚拟环境**（建议 Python 3.9+）：

    - Windows (PowerShell)：
      ```powershell
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1
      ```
    - Windows (cmd)：
      ```cmd
      python -m venv .venv
      .\.venv\Scripts\activate
      ```
    - macOS / Linux：
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

2. **使用清华镜像极速安装依赖**（已解决国内网络超时问题）：

    ```powershell
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```

3. **如需 Jupyter Lab/Notebook**（已解决 pywinpty 依赖问题，推荐用 uv 极速安装）：

    ```powershell
    pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
    uv pip install jupyter jupyterlab "pywinpty!=2.0.14" -i https://pypi.tuna.tsinghua.edu.cn/simple
    .\.venv\Scripts\jupyter lab
    ```

4. **验证安装**：

    ```powershell
    .\.venv\Scripts\jupyter --version
    ```
    若能正常显示版本号，说明环境配置成功。

> `requirements.txt` 已包含课程全部主要依赖（numpy, scipy, pandas, matplotlib, scikit-learn, sympy, seaborn, statsmodels, tensorflow, jupyter）。如需 GPU 版 TensorFlow，请参考官方文档按平台安装。

---

> **常见问题说明**：
> - 近期 `pywinpty==2.0.14` 包在 PyPI 上元数据损坏，导致 Jupyter Lab 安装失败。已通过 `pywinpty!=2.0.14` 规避。
> - 推荐使用 `uv` 包管理器极速安装依赖，兼容 pip。
> - 如遇网络超时，务必加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 环境配置与依赖安装（2026年3月最新说明）

本仓库已适配中国大陆常见网络环境，推荐如下步骤极速完成环境搭建：

1. **创建并激活虚拟环境**（如未完成）：

    - Windows (PowerShell):
      ```powershell
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1
      ```
    - Windows (cmd):
      ```cmd
      python -m venv .venv
      .\.venv\Scripts\activate
      ```
    - macOS / Linux:
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

2. **使用清华镜像极速安装依赖**（推荐，已解决国内网络超时问题）：

    ```powershell
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```

3. **如需 Jupyter Lab/Notebook**（已解决 pywinpty 依赖问题）：

    ```powershell
    pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
    uv pip install jupyter jupyterlab "pywinpty!=2.0.14" -i https://pypi.tuna.tsinghua.edu.cn/simple
    .\.venv\Scripts\jupyter lab
    ```

4. **验证安装**：

    ```powershell
    .\.venv\Scripts\jupyter --version
    ```
    若能正常显示版本号，说明环境配置成功。



## 贡献与许可
- 欢迎提交 issue 与 PR，或用作教学/自学素材。


