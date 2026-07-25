# 裂纹板材加工自研启发式求解

本项目面向“裂纹板材加工问题”。程序在共同工期 `T` 内决定每个木块不加工、普通加工或精密加工，并把加工木块分配给对应设备，使重组后的总价值尽可能高。

项目不调用 CBC、Gurobi、PuLP、OR-Tools 等包求解器。0-1 数学模型只作为目标函数、约束和算法评价口径；实际选择由自研启发式算法完成。

## 数学口径

加工时间：

```text
t_hat[u,k] = A / s_k * (1 + alpha * C_u * CS_u)
```

目标函数：

```text
F = 木块固有价值 + 普通加工增值 + 精密加工增值 + 相邻双精加工奖励
  - 精度不一致损失 - 跨块裂纹损失
```

其中 `v_u = base_value * (1 - CS_u)`。在几何裂纹输入下，`C_u`、`CS_u` 和 `L_uv` 都由裂纹折线自动计算，不在正式实验 JSON 中手写。

## 四种算法

- `VF`：单块价值优先基线。
- `VDF`：单位加工时间价值优先基线。
- `MG`：动态边际收益贪心，每轮按 `Delta F / t_hat[u,k]` 选择。
- `CNAG-LS`：Crack-and-Neighborhood-Aware Greedy Local Search，包含相邻成对精加工构造、局部搜索和规模保护。

所有方法返回统一的 `Solution`，并共享加工时间、目标函数重算、可行性检查和 CSV/JSON/图片输出。

## 安装

```bash
pip install -r requirements.txt
```

## 单实例运行

`configs/deterministic_example.json` 是一个中等复杂度确定性标准基准。

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/comparison --method all
```

也可以单独运行：

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/vf --method vf
python -m src.main --config configs/deterministic_example.json --output outputs/vdf --method vdf
python -m src.main --config configs/deterministic_example.json --output outputs/mg --method mg
python -m src.main --config configs/deterministic_example.json --output outputs/cnag_single --method cnag
```

## 正式两阶段实验

正式实验入口只有：

```text
configs/example.json
```

该文件显式写出 12 个确定性案例，每个案例包含 8 个压力情景。案例在木板规模、裂纹分布、设备结构、处理速度和工期要求上形成梯度差异。正式实验不使用随机种子、模板覆盖、情景概率、`crack_updates` 或 `edge_loss_updates`。

运行：

```bash
python -m src.experiment --config configs/example.json --output outputs/experiment
```

阶段一：每个确定性案例分别运行 `VF/VDF/MG/CNAG-LS`，保存固定方案。

阶段二：把阶段一的固定方案放入 8 个情景中评价。情景只改变设备状态和隐藏裂纹几何；程序用 `dataclasses.replace` 生成情景数据，并重新从“观测裂纹 + 隐藏裂纹”的几何折线推导 `C_u`、`CS_u`、`L_uv`。阶段二不重新求解、不重新分配。

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

## 裂纹输入

正式配置使用几何裂纹：

```json
{
  "cracks": {
    "mode": "geometry",
    "epsilon": 0.08,
    "R_max": 3.2,
    "lambda_b": 1.5,
    "items": [
      {
        "id": "OC1",
        "width": 0.36,
        "polyline": [[2.0, 12.0], [5.0, 10.8], [8.5, 9.6]]
      }
    ]
  }
}
```

每条裂纹按相邻点依次组成多段折线：`p1->p2`、`p2->p3`、`p3->p4`……。程序会校验点数、坐标范围、宽度、重复 ID、连续重复点和零长度风险。

`direct` 裂纹模式仍保留给小规模测试和向后兼容，不用于正式 `configs/example.json`。

## 测试

```bash
python -m pytest -q
```

测试覆盖四算法可行性、统一目标函数、裂纹几何推导、两阶段实验结构、阶段二不重新求解、MG 解析边际增量、CLI 输出和禁止包求解器引用。

## 当前限制

- `CNAG-LS` 是启发式局部搜索，不保证全局最优。
- 精确回溯验证器只用于极小规模实例。
- 目前只支持规则网格木块和折线裂纹几何。
- 未实现滚动调度；阶段二用于固定方案压力评估。
