from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


BlockId = str
DeviceId = str
EdgeKey = Tuple[BlockId, BlockId]
Point = Tuple[float, float]


@dataclass(frozen=True)
class Crack:
    """板材全局坐标系下的折线裂纹。"""

    id: str
    polyline: Tuple[Point, ...]
    width: float


@dataclass(frozen=True)
class Board:
    """板材与等分网格参数。"""

    width: float
    height: float
    m: int
    n: int
    base_value: float

    @property
    def area_per_block(self) -> float:
        return self.width * self.height / (self.m * self.n)


@dataclass(frozen=True)
class Block:
    """木块对象，id形如 B_1_2，i表示宽度方向列号，j表示长度方向行号。"""

    id: BlockId
    i: int
    j: int
    crack_present: int
    crack_severity: float

    @property
    def intrinsic_value_factor(self) -> float:
        return 1.0 - self.crack_severity


@dataclass(frozen=True)
class Device:
    """加工设备对象。device_type只能为 ordinary 或 precision。"""

    id: DeviceId
    device_type: str
    speed: float

    @property
    def is_ordinary(self) -> bool:
        return self.device_type == "ordinary"

    @property
    def is_precision(self) -> bool:
        return self.device_type == "precision"


@dataclass(frozen=True)
class ValueParams:
    """价值与时间影响参数。"""

    r_ordinary: float
    r_precision: float
    default_same_precision_reward: float
    default_precision_mismatch_penalty: float
    default_cross_crack_loss: float
    alpha: float


@dataclass
class ProblemData:
    """裂纹板材确定性问题所需的完整输入数据。"""

    board: Board
    deadline: float
    devices: List[Device]
    blocks: List[Block]
    edges: List[EdgeKey]
    values: ValueParams
    same_precision_reward: Dict[EdgeKey, float]
    precision_mismatch_penalty: Dict[EdgeKey, float]
    cross_crack_loss: Dict[EdgeKey, float]
    cracks: Tuple[Crack, ...] = ()
    crack_epsilon: float = 1e-6
    crack_r_max: float | None = None
    crack_lambda_b: float = 0.0

    @property
    def block_ids(self) -> List[BlockId]:
        return [block.id for block in self.blocks]

    @property
    def device_ids(self) -> List[DeviceId]:
        return [device.id for device in self.devices]

    @property
    def ordinary_device_ids(self) -> List[DeviceId]:
        return [device.id for device in self.devices if device.is_ordinary]

    @property
    def precision_device_ids(self) -> List[DeviceId]:
        return [device.id for device in self.devices if device.is_precision]

    def block_by_id(self) -> Dict[BlockId, Block]:
        return {block.id: block for block in self.blocks}

    def device_by_id(self) -> Dict[DeviceId, Device]:
        return {device.id: device for device in self.devices}


@dataclass
class Solution:
    """算法返回方案以及由方案派生的汇总信息。"""

    status: str
    objective_value: float
    solve_seconds: float
    x: Dict[Tuple[BlockId, DeviceId], int]
    y0: Dict[BlockId, int]
    y_ordinary: Dict[BlockId, int]
    y_precision: Dict[BlockId, int]
    h: Dict[EdgeKey, int]
    m: Dict[EdgeKey, int]
    solver_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def block_id(i: int, j: int) -> BlockId:
    return f"B_{i}_{j}"


def normalize_edge(u: BlockId, v: BlockId) -> EdgeKey:
    """统一相邻边顺序，避免 {u,v} 与 {v,u} 重复。"""

    return tuple(sorted((u, v)))  # type: ignore[return-value]
