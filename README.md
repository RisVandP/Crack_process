# 裂纹板材确定性0-1整数线性规划模型（BILP）

本项目实现裂纹板材加工问题的第一阶段：确定性基础模型。模型决定每个木块不加工、普通加工或精密加工，并将加工木块分配给对应类型设备，在共同工期 `T` 内最大化整块木板重组后的总价值。

## 为什么是BILP

本阶段所有决策变量均为0-1变量：

- `x[u,k]`：木块 `u` 是否分配给设备 `k`
- `y0[u]`：木块 `u` 是否不加工
- `yO[u]`：木块 `u` 是否普通加工
- `yP[u]`：木块 `u` 是否精密加工
- `h[u,v]`：相邻木块 `u,v` 是否均精密加工
- `m[u,v]`：相邻木块 `u,v` 是否精加工状态不一致

目标函数与约束均为线性形式，因此是确定性0-1整数线性规划模型（BILP）。

## 数学符号与代码对象对应

- `\mathcal U`：`ProblemData.blocks` / `data.block_ids`
- `\mathcal K`：`ProblemData.devices` / `data.device_ids`
- `\mathcal K^O`：`data.ordinary_device_ids`
- `\mathcal K^P`：`data.precision_device_ids`
- `E`：`ProblemData.edges`
- `A=WH/(mn)`：`data.board.area_per_block`
- `C_u`：`Block.crack_present`
- `CS_u`：`Block.crack_severity`
- `v_u`：`base_value * (1 - crack_severity)`
- `\hat t_{u,k}`：`processing_time(data, u, k)`
- `b_uv`：`data.same_precision_reward[edge]`
- `p_uv`：`data.precision_mismatch_penalty[edge]`
- `L_uv`：`data.cross_crack_loss[edge]`

## 安装

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/deterministic_example
```

如果不想显示CBC求解过程：

```bash
python -m src.main --config configs/deterministic_example.json --output outputs/deterministic_example --quiet
```

## 测试

```bash
pytest -q
```

测试包含：

- 自动相邻边生成检查
- 求解方案可行性检查
- 独立目标函数重算
- 2x2极小实例的暴力枚举最优值对照

## JSON输入格式

核心字段：

- `board`：板材宽度、长度、分块数和木块基准价值
- `deadline`：共同工期 `T`
- `devices`：设备编号、类型和速度
- `values`：加工增值、相邻奖励、精度不一致损失、跨块裂缝损失、裂缝时间影响系数
- `cracks`：裂缝信息，支持全局几何输入和直接输入两种方式

当前示例使用全局几何输入：

```json
"cracks": {
  "mode": "geometry",
  "epsilon": 0.001,
  "R_max": 3.0,
  "lambda_b": 3.0,
  "items": [
    {
      "id": "C1",
      "width": 0.45,
      "polyline": [[1.0, 1.0], [5.2, 3.2], [7.0, 6.5]]
    }
  ]
}
```

其中 `polyline` 是裂缝在木板全局坐标系中的折线位置，`width` 是裂缝宽度或等价破坏强度。
程序会自动计算：

- `C_u`：木块是否被裂缝经过
- `CS_u`：木块裂缝严重程度
- `L_uv`：裂缝跨越相邻木块边界造成的额外损失

为了调试和兼容，也可以使用 `mode: direct` 直接输入每个木块的 `C`、`CS`，以及每条相邻边的 `L_uv`。
相邻边参数可设置默认值，也可用 `B_1_1|B_1_2` 格式覆盖某条边。

## 输出文件

运行后输出目录包含：

- `deterministic_assignments.csv`：每个木块的裂缝状态、加工状态、设备分配、加工时间和加工增值
- `device_utilization.csv`：每台设备的任务数、总加工时间、工期和利用率
- `objective_breakdown.csv`：目标函数分项
- `summary.json`：求解状态和汇总指标
- `solver.log`：CBC求解日志
- `processing_status_grid.png`：木块加工状态图
- `device_assignment_grid.png`：木块设备分配图
- `device_utilization.png`：设备利用率图

## 当前阶段未实现

暂不实现滚动时刻、设备故障、性能下降、隐藏裂缝情景、非预见性约束、风险权重、最坏情景变量和滚动重调度模型。
