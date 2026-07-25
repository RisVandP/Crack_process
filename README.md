# 裂纹板材加工自研启发式求解

本项目面向“裂纹板材加工问题”，目标是在共同工期内决定每个木块不加工、普通加工或精密加工，并分配到对应设备，使重组后的完整价值尽可能高。

项目不再调用 CBC、Gurobi、PuLP、OR-Tools 等包求解器。原 0-1 线性模型仅作为问题形式化和目标函数依据；实际求解由自研确定性启发式算法完成。

## 数学目标

加工时间统一为：

```text
hat_t[u,k] = A / s_k * (1 + alpha * C_u * CS_u)
```

完整目标函数为：

```text
F = 固有价值
  + 普通加工增值
  + 精密加工增值
  + 相邻双精加工奖励
  - 精密状态不一致损失
  - 跨块裂纹损失
```

其中 `v_u = base_value * (1 - CS_u)`，无论木块是否加工都计入最终目标值；`L_uv` 在给定裂纹信息后是常数，也保留在目标分解中。

## 四种方法

- `VF`：单块价值优先基线，排序分数为 `v_u + A*r_q`。
- `VDF`：单位加工时间价值优先基线，排序分数为 `(v_u + A*r_q) / hat_t[u,k]`。
- `MG`：动态边际收益贪心，每轮重算 `Delta F(a|S) / hat_t[u,k]`。
- `CNAG-LS`：Crack-and-Neighborhood-Aware Greedy Local Search，裂纹-邻域感知的自适应贪心局部搜索算法。

`CNAG-LS` 包含：

1. 动态边际收益构造；
2. 相邻成对精加工候选；
3. 多邻域局部搜索，包括 `Add`、`Drop`、`ModeChange`、`Relocate`、`Swap`、`PairInsert`。

所有方法返回同一种 `Solution`，并共用加工时间、目标函数重算、可行性检查、CSV/JSON/图片输出。

## 安装

```bash
pip install -r requirements.txt
```

## 运行单个方法

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/cnag_single --method cnag
```

其他方法：

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/vf --method vf
python -m src.main --config configs/deterministic_example.json --output outputs/vdf --method vdf
python -m src.main --config configs/deterministic_example.json --output outputs/mg --method mg
```

## 运行全部启发式比较

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/comparison --method all
```

输出结构：

```text
outputs/comparison/
├── vf/
├── vdf/
├── mg/
├── cnag_ls/
├── method_comparison.csv
├── method_comparison.json
└── method_comparison.png
```

比较表中的 `difference_to_cnag_ls_percent` 表示相对 CNAG-LS 的目标差异。

## 小规模精确验证

项目提供自编 DFS 回溯验证器，仅用于极小规模实例，不调用包求解器：

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/exact --method exact
```

若规模超过限制，会返回 `not_run` 或 `limit_reached`，不会声称得到全局最优。

## 正式两阶段综合实验

正式实验唯一数据入口是：

```text
configs/example.json
```

它显式写出 12 个确定性现实组合，每个组合内部并列写出 8 个不确定压力情景。正式实验不使用随机裂纹生成、参数笛卡尔积、模板覆盖或情景概率。

运行入口：

```bash
python -m src.experiment --config configs/example.json --output outputs/experiment
```

程序分两阶段：

1. 阶段一：每个确定性组合分别运行 `VF/VDF/MG/CNAG-LS`，保存固定方案和目标分解。
2. 阶段二：同一个固定方案进入该组合的 8 个压力情景，只评价稳定性，不重新求解或重新分配。

输出包括：

- `stage1_deterministic_rows.csv/json`
- `solutions/<case_id>/<method>.json`
- `stage2_scenario_rows.csv/json`
- `method_summary.csv/json`
- `final_ranking.csv/json`
- `deterministic_by_case.png`
- `scenario_heatmap.png`
- `feasibility_comparison.png`
- `final_score_comparison.png`
- `EXPERIMENT_ANALYSIS.md`

综合得分权重从 `configs/example.json` 的 `evaluation_weights` 读取，不写死在 Python 中。

## 辅助实验入口

当前的快速调试入口仍是：

```bash
python -m src.exp_grid --config configs/exp_grid.json --output outputs/grid_exp
```

输出包括：

- `grid_rows.csv`
- `grid_sum.csv`
- `grid_sum.json`
- `obj_by_alg.png`
- `rt_by_alg.png`
- `gap_by_alg.png`
- `GRID_ANALYSIS.md`

`configs/exp_grid.json` 仅用于快速调试旧的网格实验，不作为正式实验设计入口。

## 单实例压力评价

确定性算法先基于当前观测参数生成方案；随后用多组梯度场景评价同一方案在不同参数扰动下的表现。这里的场景不带发生概率，作用是分析算法在什么条件下表现较好或较差，并探索算法适用范围：

```bash
python -m src.scn_eval --config configs/scn_grid.json --output outputs/scn_eval --method all
```

压力测试配置文件中，每个等级只描述对已有确定性输入的修改，例如移除某台设备、降低设备速度、更新隐藏裂纹严重程度和跨块裂纹损失。示例配置采用 `S0-基准`、`S1-轻度扰动`、`S2-中度扰动`、`S3-重度扰动` 的梯度设计，确定性求解逻辑不因压力评价而改变。

`configs/scn_grid.json` 和 `src.scn_eval` 是单实例兼容入口；正式 12 组合 × 8 情景实验请使用 `configs/example.json` 和 `src.experiment`。

输出包括：

- `scn_rows.csv`
- `scn_sum.csv`
- `scn_sum.json`
- `scn_plot.png`

## JSON 输入

核心字段：

- `board`：木板宽度、长度、分块数、基准价值；
- `deadline`：共同工期；
- `devices`：设备编号、类型、速度；
- `values`：加工增值、相邻奖励、精度不一致损失、裂纹时间影响系数；
- `cracks`：裂纹信息，支持几何折线输入和直接输入。

示例配置在 `configs/deterministic_example.json`。

## 测试

```bash
python -m pytest -q
```

测试覆盖：

- 四种启发式可行性；
- 统一目标函数重算；
- 加工时间裂纹修正；
- `h_uv` 与 `m_uv` 逻辑；
- 小规模精确回溯；
- CLI 端到端输出；
- 源码和依赖中不存在包求解器调用。

## 当前限制

- CNAG-LS 是启发式局部搜索，不保证全局最优；
- 精确回溯只用于极小规模验证；
- 裂纹几何当前支持折线与规则网格；
- 未实现滚动调度；
- 大规模场景可进一步加入增量缓存、候选剪枝和并行边际收益计算。
