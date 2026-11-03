# RETHINK：四项分析实验的图、原始数据与脚本

本仓库按四个实验分目录，收录现有图、逐决策／逐 token 原始测量、数据表、指标定义和离线重绘脚本。

## 四项实验一览

| 文件夹 | 要回答的问题 | 主要结果 |
| --- | --- | --- |
| [01：基线信号分布](01_baseline_signal_distribution/) | 基线中 Reasoning 和 Action 的 Teacher–Student 信号差异有多大？ | Action 均值为 0.0346 ；Reasoning均值为 0.2340 |
| [02：Reasoning 条件干预](02_reasoning_intervention/) | 在相同历史下，正确／删除／错误 Reasoning 怎样影响下一步动作？ | 正确 R：19/22、17/22；删除 R：13/22、13/22；错误 R：10/22、9/22 |
| [03：RETHINK 插入干预](03_rethink_insertion/) | 固定 H/R，只插入一个 marker，后续动作的正确概率如何变化？ | 正确动作平均概率 Q：0.1208 → 0.6119 |
| [04：三种触发模式](04_bad_reasoning_action_quality/) | 禁止／强制／自主触发rethink标记时，生成正确动作的比例怎样变化？ | 禁止：17/208、强制：164/208、自主：64/208|

## 01 · 基线 Reasoning / Action 信号分布

![基线信号分布](01_baseline_signal_distribution/figures/figure.png)

测量了每个决策内部的平均绝对 Teacher–Student sampled-token log-probability gap，紫色是 Action，灰色是 Reasoning，虚线是均值，实线是平滑后的分箱概率，y 轴断开处已标出。由图可见，action的平均信号远远弱于reasoning的平均信号，用于直观展现motivation。

## 02 · 相同 History 下改变 Reasoning

![Reasoning 条件对比](02_reasoning_intervention/figures/figure.png)

外环为 Teacher 模型、内环为 Student 模型，蓝色表示下一步动作属于正确动作，紫色表示动作不是正确动作，分为三种情况：提供正确的reasoning前缀，提供错误的reasoning前缀和删除reasoning前缀，用于展现教师的额外信息能造成的改变远小于reasoning对后续的影响，辅助说明reasoning对action的lock-in效果。

## 03 · 插入rethink标记对动作正确率的影响

![真实1000对RETHINK干预](03_rethink_insertion/figures/figure.png)

抽取了1000条bad reasoning作为前缀，在同样的前缀下分为插入rethink标记与不插入rethink标记，共2000个样本点，让 ours 方法的模型进行自由生成，比较插入rethink标记后的动作正确率，用于说明rethink标记的确解除了reasoning对action的lock-in效果，起到了纠错的作用。

## 04 · 禁止触发，强制触发rethink标记与自由生成时的动作正确率比较

![三种触发模式](04_bad_reasoning_action_quality/figures/figure.png)

在固定 Reasoning 后分别禁止 rethink标记、强制插入 rethink标记和自由生成。蓝色为正确动作的概率，斜线紫色是不正确动作的概率，用于展现插入rethink标记对后续动作正确率影响的另一种表达形式（和第三个有点重复，图表形式不同）

## 仓库内有什么

每个实验目录均含 `README.md`、`figures/`、`data/`、`raw/` 和 `plot.py`。`data/` 是直接用于画图的数据表，`raw/` 保留用于重算这些表的逐条原始测量与冻结输入／审核记录。原始科学内容按字节保留，`artifact_manifest.json` 记录逐文件来源指纹，不含私人操作路径。

图②、图④的展示图保留已提供的原图；对应 `figures/reproduced.png` / `.svg` 和 `plot.py` 是**相同数据的重绘版**，不声称恢复原图的精确排版源码。图①、图③附有原始 PNG/SVG；重绘可能因系统字体产生少量排版差异。所有重绘默认另存，不覆盖原图和数据。

## 独立离线运行

只需要 **Python 3.11+、requirements.txt 里的 NumPy / Matplotlib，以及一个中文字体**。Python 3.13.11 是已用的验证版本。不需要 GPU、模型权重、API key、原工作区或原服务器。

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell 使用：.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python verify_data.py
python plot_all.py
```

Linux 若没有中文字体，先安装系统字体包（Debian/Ubuntu 示例）：

```bash
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk
```

Windows 可使用已有的 Microsoft YaHei / SimHei；Linux 优先 Noto Sans CJK SC。字体是明确的系统依赖，没有分发商业字体文件。预生成 PNG 和 SVG 可直接查看，无需安装任何东西。

单独重绘某项：

```bash
python 03_rethink_insertion/plot.py --output-dir reproduced/03_rethink_insertion
```

`verify_data.py` 仅使用 Python 标准库，逐文件校验 `SHA256SUMS.txt`，从原始记录重算四项数字、配对关系、候选概率和监督边界。`plot_all.py` 离线重绘四项到 `reproduced/`。

实际的干净检出、全新虚拟环境安装与四项重绘结果见 [离线复现验证记录](REPRODUCTION.md)。

**复现范围：**这里验证的是“仓库内原始测量 → 数值核验 → 图”。不包含模型权重，不声称仅凭本仓库可重新执行原始 GPU 训练／推理实验。ALFWorld 环境观察、动作、静态 skill 和记录的审核属于本次分析数据；没有上传账号、SSH、API 密钥、机器配置或无关训练日志。

许可未另行指定；本次先按私有研究资料整理，公开和再分发前请核对原始数据与上游资产的适用许可。
