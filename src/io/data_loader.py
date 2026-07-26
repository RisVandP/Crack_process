from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from ..model.data_models import (
    Block,
    Board,
    Device,
    EdgeKey,
    ProblemData,
    ValueParams,
    block_id,
    normalize_edge,
    Crack,
)
from ..geometry.crack_geometry import compute_crack_effects


def load_problem(path: str | Path) -> ProblemData:
    """从JSON文件读取确定性问题实例，并自动生成木块与相邻边。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_problem(raw)


def parse_problem(raw: Dict[str, Any]) -> ProblemData:
    board_raw = raw["board"]
    board = Board(
        width=float(board_raw["width"]),
        height=float(board_raw["height"]),
        m=int(board_raw["m"]),
        n=int(board_raw["n"]),
        base_value=float(board_raw["base_value"]),
    )
    if board.m <= 0 or board.n <= 0:
        raise ValueError("m和n必须为正整数。")
    if board.width <= 0 or board.height <= 0:
        raise ValueError("板材宽度和长度必须为正数。")

    value_raw = raw["values"]
    values = ValueParams(
        r_ordinary=float(value_raw["r_ordinary"]),
        r_precision=float(value_raw["r_precision"]),
        default_same_precision_reward=float(value_raw.get("default_same_precision_reward", 0.0)),
        default_precision_mismatch_penalty=float(value_raw.get("default_precision_mismatch_penalty", 0.0)),
        default_cross_crack_loss=float(value_raw.get("default_cross_crack_loss", 0.0)),
        alpha=float(value_raw.get("alpha", 0.0)),
    )

    devices = [
        Device(
            id=str(item["id"]),
            device_type=str(item["type"]),
            speed=float(item["speed"]),
        )
        for item in raw["devices"]
    ]
    _validate_devices(devices)

    edges = _build_edges(board)
    crack_by_block, geometry_cross_loss, cracks, crack_epsilon, crack_r_max, crack_lambda_b = _read_cracks(raw, board, edges)
    blocks = _build_blocks(board, crack_by_block)

    same_reward = _edge_values(edges, values.default_same_precision_reward, value_raw.get("same_precision_reward", {}))
    mismatch_penalty = _edge_values(edges, values.default_precision_mismatch_penalty, value_raw.get("precision_mismatch_penalty", {}))
    if geometry_cross_loss is None:
        cross_loss = _edge_values(edges, values.default_cross_crack_loss, value_raw.get("cross_crack_loss", {}))
    else:
        cross_loss = geometry_cross_loss
    block_intrinsic_value = {block.id: block.intrinsic_value for block in blocks}
    intrinsic_block_value = sum(block_intrinsic_value.values())
    cross_crack_loss_total = sum(cross_loss[edge] for edge in edges)

    deadline = float(raw["deadline"])
    if deadline <= 0:
        raise ValueError("共同工期deadline必须为正数。")

    return ProblemData(
        board=board,
        deadline=deadline,
        devices=devices,
        blocks=blocks,
        edges=edges,
        values=values,
        same_precision_reward=same_reward,
        precision_mismatch_penalty=mismatch_penalty,
        cross_crack_loss=cross_loss,
        block_intrinsic_value=block_intrinsic_value,
        intrinsic_block_value=intrinsic_block_value,
        cross_crack_loss_total=cross_crack_loss_total,
        cracks=tuple(cracks),
        crack_epsilon=crack_epsilon,
        crack_r_max=crack_r_max,
        crack_lambda_b=crack_lambda_b,
    )


def _read_cracks(raw: Dict[str, Any], board: Board, edges: list[EdgeKey]) -> tuple[Dict[str, Dict[str, Any]], Dict[EdgeKey, float] | None, list[Crack], float, float | None, float]:
    """读取裂缝信息。

    支持两种模式：
    1. direct：直接输入每个木块的C、CS和每条边的L_uv；
    2. geometry：输入全局裂缝折线与宽度，由程序计算C、CS和L_uv。
    """

    cracks_raw = raw.get("cracks", {})
    mode = cracks_raw.get("mode", "direct")
    if mode == "direct":
        return cracks_raw.get("blocks", {}), None, [], 1e-6, None, 0.0
    if mode != "geometry":
        raise ValueError("cracks.mode必须为 direct 或 geometry。")

    epsilon = float(cracks_raw.get("epsilon", 1e-6))
    r_max_raw = cracks_raw.get("R_max")
    r_max = None if r_max_raw is None else float(r_max_raw)
    lambda_b = float(cracks_raw.get("lambda_b", raw["values"].get("default_cross_crack_loss", 0.0)))
    crack_items = parse_crack_items(cracks_raw.get("items", []), board)
    block_cracks, edge_loss = compute_crack_effects(board, edges, crack_items, epsilon, r_max, lambda_b)
    return block_cracks, edge_loss, crack_items, epsilon, r_max, lambda_b


def parse_crack_items(items, board: Board) -> list[Crack]:
    """解析并校验几何裂纹折线。"""

    cracks: list[Crack] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        crack_id = str(item.get("id", f"C{index}"))
        if crack_id in seen:
            raise ValueError(f"裂缝ID重复：{crack_id}")
        seen.add(crack_id)
        width = float(item["width"])
        if width <= 0:
            raise ValueError(f"裂缝 {crack_id} 的width必须大于0。")
        polyline = []
        for raw_point in item["polyline"]:
            if len(raw_point) != 2:
                raise ValueError(f"裂缝 {crack_id} 的坐标点格式必须为[x,y]。")
            x, y = float(raw_point[0]), float(raw_point[1])
            if not 0.0 <= x <= board.width or not 0.0 <= y <= board.height:
                raise ValueError(f"裂缝 {crack_id} 的坐标({x},{y})超出板材范围。")
            if polyline and polyline[-1] == (x, y):
                raise ValueError(f"裂缝 {crack_id} 存在连续重复点。")
            polyline.append((x, y))
        if len(polyline) < 2:
            raise ValueError(f"裂缝 {crack_id} 至少需要两个点。")
        cracks.append(Crack(id=crack_id, polyline=tuple(polyline), width=width))
    return cracks


def _validate_devices(devices: Iterable[Device]) -> None:
    seen = set()
    count = 0
    for device in devices:
        count += 1
        if device.id in seen:
            raise ValueError(f"设备编号重复：{device.id}")
        seen.add(device.id)
        if device.device_type not in {"ordinary", "precision"}:
            raise ValueError(f"设备 {device.id} 的type必须是 ordinary 或 precision。")
        if device.speed <= 0:
            raise ValueError(f"设备 {device.id} 的speed必须为正数。")
    if count == 0:
        raise ValueError("至少需要提供一台设备。")


def _build_blocks(board: Board, crack_by_block: Dict[str, Dict[str, Any]]) -> list[Block]:
    blocks: list[Block] = []
    for i in range(1, board.m + 1):
        for j in range(1, board.n + 1):
            bid = block_id(i, j)
            crack_info = crack_by_block.get(bid, {})
            crack_present = int(crack_info.get("C", 0))
            crack_severity = float(crack_info.get("CS", 0.0))
            if crack_present not in {0, 1}:
                raise ValueError(f"{bid} 的 C 必须为0或1。")
            if not 0.0 <= crack_severity <= 1.0:
                raise ValueError(f"{bid} 的 CS 必须位于[0,1]。")
            if crack_present == 0 and crack_severity > 0:
                raise ValueError(f"{bid} 的 C=0 但 CS>0，裂缝状态不一致。")
            intrinsic_value = board.base_value * (1.0 - crack_severity)
            blocks.append(Block(bid, i, j, crack_present, crack_severity, intrinsic_value))
    return blocks


def _build_edges(board: Board) -> list[EdgeKey]:
    """生成上、下、左、右相邻边，斜对角不算相邻。"""

    edges: list[EdgeKey] = []
    for i in range(1, board.m + 1):
        for j in range(1, board.n + 1):
            current = block_id(i, j)
            if i < board.m:
                edges.append(normalize_edge(current, block_id(i + 1, j)))
            if j < board.n:
                edges.append(normalize_edge(current, block_id(i, j + 1)))
    return sorted(set(edges))


def _edge_values(edges: list[EdgeKey], default: float, overrides: Dict[str, Any]) -> Dict[EdgeKey, float]:
    values = {edge: float(default) for edge in edges}
    for raw_key, raw_value in overrides.items():
        edge = parse_edge_key(raw_key)
        if edge not in values:
            raise ValueError(f"边 {raw_key} 不是合法相邻边。")
        values[edge] = float(raw_value)
    return values


def parse_edge_key(raw_key: str) -> EdgeKey:
    """解析JSON中的边键，格式为 B_1_1|B_1_2。"""

    parts = raw_key.split("|")
    if len(parts) != 2:
        raise ValueError(f"边键格式错误：{raw_key}，应为 B_i_j|B_i_j。")
    return normalize_edge(parts[0], parts[1])


def edge_key_to_str(edge: Tuple[str, str]) -> str:
    return f"{edge[0]}|{edge[1]}"
