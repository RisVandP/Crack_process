# 实验数据设计说明

本项目的实验输入分为两类，对应两个分析目标。

## 1. 参数影响规律分析

配置文件：`configs/sensitivity.json`

目的：分析木板规模、裂缝分布、设备性能、加工时间和价值结构变化对最终结果的整体影响趋势。

设计方式分为两部分：

| 类型 | Case ID | 含义 |
|---|---|---|
| 单因素梯度 | `SCALE01`-`SCALE06` | 分块数量从 12 块逐步增加到 60 块 |
| 单因素梯度 | `CRACK01`-`CRACK06` | 裂缝数量从 0 条逐步增加到 5 条 |
| 单因素梯度 | `DEVICE01`-`DEVICE06` | 普通与精密设备速度按比例从低到高变化 |
| 单因素梯度 | `TIME01`-`TIME06` | 工期 `T` 从极紧逐步放宽 |
| 单因素梯度 | `VALUE01`-`VALUE06` | 改变普通增值、精密增值、相邻奖励和不一致惩罚组合 |
| 组合梯度 | `COMBO01`-`COMBO06` | 同时改变木板规模、裂缝分布、设备性能和工期，形成从容易到困难的综合输入组合 |

其中 `COMBO01`-`COMBO06` 用来回应题目中“不同组合对结果的影响”的要求：

| Case ID | 组合特征 |
|---|---|
| `COMBO01` | 小规模、无裂缝、高设备性能、宽松工期 |
| `COMBO02` | 中小规模、轻裂缝、较高设备性能、较宽松工期 |
| `COMBO03` | 中等规模、中等裂缝、中等偏高设备性能、正常工期 |
| `COMBO04` | 中大规模、较多裂缝、中等设备性能、偏紧工期 |
| `COMBO05` | 大规模、多裂缝、较低设备性能、紧工期 |
| `COMBO06` | 超大规模、密集裂缝、低设备性能、极紧工期 |

运行命令：

```bash
python -m src.experiments.experiment --config configs/sensitivity.json --output outputs/sensitivity_check
```

主要输出：

- `stage1_deterministic_rows.csv`：36 组 case × 4 个算法的确定性结果。
- `stage2_scenario_rows.csv`：36 组 case × 8 个情景 × 4 个算法的情景评价。
- `obj_overview.png`、`decision_overview.png`：不同因素整体对比。
- `scale_obj.png`、`crack_obj.png`、`device_obj.png`、`time_obj.png`、`value_obj.png`、`combo_obj.png`：各因素目标值趋势图。
- `scale_dec.png`、`crack_dec.png`、`device_dec.png`、`time_dec.png`、`value_dec.png`、`combo_dec.png`：各因素决策贡献趋势图。

## 2. 算法适用性场景实验

配置文件：`configs/example.json`

目的：构造典型综合场景，比较 VF、VDF、MG、MG-LS 分别适用于哪些条件。

设计方式：保留 12 组典型 case，每组强调一种实际场景或算法差异来源。

| Case ID | 场景定位 |
|---|---|
| `C01` | 均衡基准 |
| `C02` | 小规模时间密度主导 |
| `C03` | 大规模相邻关系主导 |
| `C04` | 大规模紧工期资源竞争 |
| `C05` | 裂纹耗时差异明显 |
| `C06` | 严重裂纹与相邻损失 |
| `C07` | 宽幅板局部搜索收益 |
| `C08` | 精密设备瓶颈与普通产能 |
| `C09` | 设备速度非均衡 |
| `C10` | 小规模极紧资源 |
| `C11` | 精密设备增加与局部修正 |
| `C12` | 大规模严重瓶颈 |

运行命令：

```bash
python -m src.experiments.experiment --config configs/example.json --output outputs/exp_retuned_params
```

主要输出：

- `objective_by_case_line.png`：12 个典型场景下四种算法的总价值曲线。
- `decision_value_by_case_line.png`：去掉共同基线后的决策贡献曲线。
- `method_summary.csv` 和 `final_ranking.csv`：综合考虑确定性质量、情景鲁棒性、可行率和运行时间后的算法表现。

## 3. PPT 使用建议

建议先讲 `sensitivity.json` 的参数影响规律分析：用组合梯度说明不同输入组合会显著影响结果，再用单因素梯度解释影响来源。

之后再讲 `example.json` 的算法适用性场景实验：说明不同算法在不同典型场景下的优势和不足。
