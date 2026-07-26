# 实验数据设计说明

本项目的实验输入分为两类，对应两种不同的分析目标。

## 1. 参数敏感性实验

配置文件：`configs/sensitivity.json`

目的：分析木板规模、裂缝分布、设备性能、加工时间和价值结构变化对最终结果的整体影响趋势。

设计方式：控制变量。每类因素设置 6 个梯度，共 30 组确定性案例；每组案例仍保留 8 个不确定情景，便于后续统一评估鲁棒性。

| 因素 | Case ID | 梯度含义 |
|---|---|---|
| 木板规模 | `SCALE01`-`SCALE06` | 分块数量从 12 块逐步增加到 60 块 |
| 裂缝分布 | `CRACK01`-`CRACK06` | 裂纹数量从 0 条逐步增加到 5 条 |
| 设备性能 | `DEVICE01`-`DEVICE06` | 普通与精密设备速度按比例从低到高变化 |
| 加工时间 | `TIME01`-`TIME06` | 工期 `T` 从极紧逐步放宽 |
| 价值结构 | `VALUE01`-`VALUE06` | 改变普通/精密增值、相邻奖励和不一致惩罚组合 |

运行命令：

```bash
python -m src.experiments.experiment --config configs/sensitivity.json --output outputs/sensitivity_check
```

主要输出：

- `stage1_deterministic_rows.csv`：30 组案例 × 4 个算法的确定性结果。
- `stage2_scenario_rows.csv`：30 组案例 × 8 个情景 × 4 个算法的情景评估。
- `sensitivity_*_objective.png`：每类因素的总价值趋势图。
- `sensitivity_*_decision.png`：每类因素的决策贡献趋势图。

## 2. 算法适用性场景实验

配置文件：`configs/example.json`

目的：构造典型组合场景，比较 VF、VDF、MG、MG-LS 分别适用于哪些条件。

设计方式：多因素组合。保留 12 组典型案例，每组案例强调一种实际场景或算法差异来源。

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

## 3. 使用建议

PPT 中建议先讲参数敏感性实验，说明输入条件变化会如何影响总体价值趋势；再讲算法适用性实验，说明不同算法在不同典型场景下的优劣。
