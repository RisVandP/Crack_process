from __future__ import annotations

from math import hypot
from typing import Dict, Iterable, List, Tuple

from .data_models import Board, Crack, EdgeKey, Point, block_id


Rect = Tuple[float, float, float, float]


def block_rect(board: Board, i: int, j: int) -> Rect:
    """返回木块的矩形区域：(xmin, xmax, ymin, ymax)。"""

    xmin = (i - 1) * board.width / board.m
    xmax = i * board.width / board.m
    ymin = (j - 1) * board.height / board.n
    ymax = j * board.height / board.n
    return xmin, xmax, ymin, ymax


def compute_crack_effects(
    board: Board,
    edges: List[EdgeKey],
    cracks: Iterable[Crack],
    epsilon: float,
    r_max: float | None,
    lambda_b: float,
) -> tuple[Dict[str, Dict[str, float]], Dict[EdgeKey, float]]:
    """由全局裂缝几何计算每个木块的C_u、CS_u和每条相邻边的L_uv。

    严重程度计算：
        R_u = sum_c width_c * ell_cu
        CS_u = min(1, R_u / R_max)

    跨块裂缝损失：
        L_uv = lambda_b * 跨越公共边界的裂缝条数
    """

    block_lengths: Dict[str, float] = {
        block_id(i, j): 0.0 for i in range(1, board.m + 1) for j in range(1, board.n + 1)
    }
    raw_damage: Dict[str, float] = dict(block_lengths)
    cross_count: Dict[EdgeKey, int] = {edge: 0 for edge in edges}

    crack_list = list(cracks)
    for crack in crack_list:
        per_block_length: Dict[str, float] = {}
        for i in range(1, board.m + 1):
            for j in range(1, board.n + 1):
                bid = block_id(i, j)
                length = polyline_length_inside_rect(crack.polyline, block_rect(board, i, j))
                per_block_length[bid] = length
                if length > epsilon:
                    block_lengths[bid] += length
                    raw_damage[bid] += crack.width * length

        for edge in edges:
            u, v = edge
            if per_block_length[u] > epsilon and per_block_length[v] > epsilon and crosses_shared_boundary(board, edge, crack.polyline):
                cross_count[edge] += 1

    if r_max is None:
        max_damage = max(raw_damage.values(), default=0.0)
        # 若实例没有裂缝，取1避免除零，此时所有CS仍为0。
        normalizer = max(max_damage, 1.0)
    else:
        normalizer = r_max
    if normalizer <= 0:
        raise ValueError("R_max必须为正数。")

    block_cracks = {}
    for bid, damage in raw_damage.items():
        cs = min(1.0, damage / normalizer)
        block_cracks[bid] = {"C": 1.0 if block_lengths[bid] > epsilon else 0.0, "CS": cs}

    edge_loss = {edge: lambda_b * count for edge, count in cross_count.items()}
    return block_cracks, edge_loss


def polyline_length_inside_rect(polyline: List[Point], rect: Rect) -> float:
    total = 0.0
    for p0, p1 in zip(polyline, polyline[1:]):
        clipped = clip_segment_to_rect(p0, p1, rect)
        if clipped is not None:
            (x0, y0), (x1, y1) = clipped
            total += hypot(x1 - x0, y1 - y0)
    return total


def clip_segment_to_rect(p0: Point, p1: Point, rect: Rect) -> tuple[Point, Point] | None:
    """Liang-Barsky线段裁剪，返回线段在矩形内的部分。"""

    xmin, xmax, ymin, ymax = rect
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0

    for p, q in [(-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def crosses_shared_boundary(board: Board, edge: EdgeKey, polyline: List[Point]) -> bool:
    """判断裂缝折线是否穿过相邻木块公共边界的相对内部。"""

    u, v = edge
    ui, uj = _parse_block_id(u)
    vi, vj = _parse_block_id(v)
    if abs(ui - vi) + abs(uj - vj) != 1:
        return False

    if ui != vi:
        # 左右相邻，公共边界是竖线x=const。
        boundary_x = max(ui, vi) - 1
        x = boundary_x * board.width / board.m
        y_low = (uj - 1) * board.height / board.n
        y_high = uj * board.height / board.n
        return any(_segment_crosses_vertical(p0, p1, x, y_low, y_high) for p0, p1 in zip(polyline, polyline[1:]))

    # 上下相邻，公共边界是横线y=const。
    boundary_y = max(uj, vj) - 1
    y = boundary_y * board.height / board.n
    x_low = (ui - 1) * board.width / board.m
    x_high = ui * board.width / board.m
    return any(_segment_crosses_horizontal(p0, p1, y, x_low, x_high) for p0, p1 in zip(polyline, polyline[1:]))


def _segment_crosses_vertical(p0: Point, p1: Point, x: float, y_low: float, y_high: float) -> bool:
    x0, y0 = p0
    x1, y1 = p1
    if (x0 - x) * (x1 - x) > 0 or abs(x1 - x0) < 1e-12:
        return False
    t = (x - x0) / (x1 - x0)
    if not 0.0 < t < 1.0:
        return False
    y = y0 + t * (y1 - y0)
    return y_low < y < y_high


def _segment_crosses_horizontal(p0: Point, p1: Point, y: float, x_low: float, x_high: float) -> bool:
    x0, y0 = p0
    x1, y1 = p1
    if (y0 - y) * (y1 - y) > 0 or abs(y1 - y0) < 1e-12:
        return False
    t = (y - y0) / (y1 - y0)
    if not 0.0 < t < 1.0:
        return False
    x = x0 + t * (x1 - x0)
    return x_low < x < x_high


def _parse_block_id(raw: str) -> tuple[int, int]:
    _, i, j = raw.split("_")
    return int(i), int(j)
