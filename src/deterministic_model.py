from __future__ import annotations

from .data_models import BlockId, DeviceId, ProblemData


def processing_time(data: ProblemData, block_id: BlockId, device_id: DeviceId) -> float:
    """统一加工时间公式。

    本项目不再调用包求解器；该文件仅保留数学模型中的加工时间计算，
    供所有启发式算法、可行性检查器和输出模块共用。
    """

    block = data.block_by_id()[block_id]
    device = data.device_by_id()[device_id]
    area = data.board.area_per_block
    return area / device.speed * (1.0 + data.values.alpha * block.crack_present * block.crack_severity)
