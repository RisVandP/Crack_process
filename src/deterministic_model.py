from __future__ import annotations

import time
from typing import Dict, Tuple

import pulp

from .data_models import BlockId, DeviceId, EdgeKey, ProblemData, Solution


def processing_time(data: ProblemData, block_id: BlockId, device_id: DeviceId) -> float:
    """计算 hat_t[u,k] = A/s_k * (1 + alpha*C_u*CS_u)。"""

    block = data.block_by_id()[block_id]
    device = data.device_by_id()[device_id]
    area = data.board.area_per_block
    return area / device.speed * (1.0 + data.values.alpha * block.crack_present * block.crack_severity)


def build_deterministic_bilp(data: ProblemData) -> tuple[pulp.LpProblem, dict]:
    """构建确定性0-1整数线性规划模型。"""

    model = pulp.LpProblem("cracked_board_deterministic_BILP", pulp.LpMaximize)
    U = data.block_ids
    K = data.device_ids
    KO = data.ordinary_device_ids
    KP = data.precision_device_ids
    E = data.edges

    # 决策变量：x[u,k]表示木块u是否分配给设备k。
    x = pulp.LpVariable.dicts("x", (U, K), lowBound=0, upBound=1, cat=pulp.LpBinary)
    y0 = pulp.LpVariable.dicts("y0", U, lowBound=0, upBound=1, cat=pulp.LpBinary)
    yO = pulp.LpVariable.dicts("yO", U, lowBound=0, upBound=1, cat=pulp.LpBinary)
    yP = pulp.LpVariable.dicts("yP", U, lowBound=0, upBound=1, cat=pulp.LpBinary)
    h = pulp.LpVariable.dicts("h", E, lowBound=0, upBound=1, cat=pulp.LpBinary)
    m = pulp.LpVariable.dicts("m", E, lowBound=0, upBound=1, cat=pulp.LpBinary)

    # 状态与设备分配对应关系。
    for u in U:
        model += y0[u] + yO[u] + yP[u] == 1, f"one_status_{u}"
        model += pulp.lpSum(x[u][k] for k in K) <= 1, f"at_most_one_device_{u}"
        model += pulp.lpSum(x[u][k] for k in KO) == yO[u], f"ordinary_assignment_{u}"
        model += pulp.lpSum(x[u][k] for k in KP) == yP[u], f"precision_assignment_{u}"

    # 共同工期约束：每台设备串行加工，所有设备并行，单台设备总时间不能超过T。
    for k in K:
        model += (
            pulp.lpSum(processing_time(data, u, k) * x[u][k] for u in U) <= data.deadline
        ), f"deadline_{k}"

    # h_uv线性化：h=1当且仅当相邻两块都精密加工。
    for edge in E:
        u, v = edge
        model += h[edge] <= yP[u], f"h_upper_u_{u}_{v}"
        model += h[edge] <= yP[v], f"h_upper_v_{u}_{v}"
        model += h[edge] >= yP[u] + yP[v] - 1, f"h_lower_{u}_{v}"
        model += m[edge] == yP[u] + yP[v] - 2 * h[edge], f"m_xor_{u}_{v}"

    block_map = data.block_by_id()
    area = data.board.area_per_block
    intrinsic_value = pulp.lpSum(
        data.board.base_value * block_map[u].intrinsic_value_factor for u in U
    )
    proc_value = pulp.lpSum(area * data.values.r_ordinary * yO[u] + area * data.values.r_precision * yP[u] for u in U)
    same_value = pulp.lpSum(data.same_precision_reward[e] * h[e] for e in E)
    diff_loss = pulp.lpSum(data.precision_mismatch_penalty[e] * m[e] for e in E)
    crack_loss = sum(data.cross_crack_loss[e] for e in E)

    model += intrinsic_value + proc_value + same_value - diff_loss - crack_loss

    variables = {"x": x, "y0": y0, "yO": yO, "yP": yP, "h": h, "m": m}
    return model, variables


def solve_deterministic(data: ProblemData, msg: bool = True, log_path: str | None = None) -> Solution:
    """调用PuLP+CBC求解确定性BILP。"""

    model, variables = build_deterministic_bilp(data)
    solver = pulp.PULP_CBC_CMD(msg=msg, logPath=log_path)
    start = time.perf_counter()
    model.solve(solver)
    elapsed = time.perf_counter() - start

    def binary_value(var) -> int:
        value = var.value()
        return int(round(value or 0))

    x_values: Dict[Tuple[BlockId, DeviceId], int] = {}
    for u in data.block_ids:
        for k in data.device_ids:
            x_values[(u, k)] = binary_value(variables["x"][u][k])

    solution = Solution(
        status=pulp.LpStatus[model.status],
        objective_value=float(pulp.value(model.objective)),
        solve_seconds=elapsed,
        x=x_values,
        y0={u: binary_value(variables["y0"][u]) for u in data.block_ids},
        y_ordinary={u: binary_value(variables["yO"][u]) for u in data.block_ids},
        y_precision={u: binary_value(variables["yP"][u]) for u in data.block_ids},
        h={e: binary_value(variables["h"][e]) for e in data.edges},
        m={e: binary_value(variables["m"][e]) for e in data.edges},
        solver_name="PuLP_CBC",
    )
    return solution
