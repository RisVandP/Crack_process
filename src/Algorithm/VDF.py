from __future__ import annotations

from ..model.data_models import ProblemData, Solution
from .VF import _density_score, _solve_static_greedy


def solve_value_density_first(data: ProblemData) -> Solution:
    # 按单位加工时间价值排序，构造VDF基线方案。
    """单位加工时间价值优先基线：按 (v_u + A*r_q)/hat_t[u,k] 排序。"""

    return _solve_static_greedy(data, "VDF", _density_score)
