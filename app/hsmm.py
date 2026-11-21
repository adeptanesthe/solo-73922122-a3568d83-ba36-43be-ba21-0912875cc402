"""联合推断两台二进制设备开停状态的概率模型。

模型为显式时长隐半马尔可夫模型 (HSMM)：

* 隐状态为两台设备状态的笛卡尔积 (s1, s2)，各自带一个“当前状态已持续分钟数”
  计数器 (c1, c2)，计数器达到上限后封顶，从而保证任意开/停段长度都可表达。
* 只有在当前状态持续时间达到该状态的最短时长后，设备才允许翻转；
  翻转后计数器重置为 1，否则计数器递增（封顶）。
* 观测为两台设备各自功率之和叠加高斯噪声 N(0, sigma^2)；
  缺测分钟不产生观测似然（对所有状态似然为 1），但状态转移照常进行，
  因此缺测分钟仍计入开/停时长。
* 允许状态在观测窗口首尾截断：初始分布对所有增强状态均匀，
  不要求第一个状态已经持续满最短时长。

推理使用对数空间前向-后向算法，得到每个分钟各设备处于开机状态的
后验概率，再按置信度阈值映射为 running / stopped / uncertain。

转移按入度/出度分组成紧凑数组，绝大多数状态只有 1 个前驱/后继，
从而让全天 1440 分钟的推断保持在秒级。

内存实现（分块检查点前向-后向，精确、无近似）
------------------------------------------------
朴素实现需要保存完整的 (T, S) alpha 与 beta 矩阵，峰值内存随窗口长度
线性增长（T=1440、S=3600 时仅这两个矩阵就约 160MB）。本实现把时间轴
切成固定大小的块（默认 128 分钟）：

1. 前向只保留当前块的 alpha 行；每个块边界把当时的 alpha 向量
   （S 个 float64）存入检查点，检查点数量为 ceil(T/B)-1，与 T 无关地
   由块大小决定单块工作集；
2. 后向逐块回放：用边界检查点作为种子，在块内重新前向一次以重建该块
   的 alpha（重算量恰好 1 倍，属于精确重算，不做任何近似/降采样），
   同时反向递推 beta，块内逐分钟计算 gamma 与开机边际概率，
   不保留任何完整窗口长度的 (T, S) 矩阵。

因此峰值工作内存为 O(B*S + (T/B)*S + S)，由固定块大小 B 和状态数 S
决定，而不是 O(T*S)。所有浮点运算的顺序与朴素实现逐行一致，
结果位级等价（仅受浮点结合律影响，不存在算法性误差）。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

NEG_INF = -math.inf
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# 前向-后向分块大小（分钟）。峰值工作内存主要由它与状态数决定，
# 与窗口总长度无关；块越小检查点越多（重算量不变，仍是 1 倍）。
DEFAULT_BLOCK_SIZE = 128
# 编译模型缓存的最大条目数，避免不同参数组合的请求无限累积。
_MODEL_CACHE_MAX = 32


@dataclass(frozen=True)
class DeviceParams:
    """单台设备的功率与最短时长参数。"""

    on_power: float
    off_power: float
    min_on: int
    min_off: int


@dataclass
class _Model:
    """编译后的增强状态空间与分组转移结构。

    pred_groups / succ_groups: 键为入度/出度 d，值为
    (目标状态数组, 对应的 d 个邻居索引数组, 对应对数权重数组)。
    """

    n_states: int
    k1: int
    k2: int
    state_s1: np.ndarray
    state_s2: np.ndarray
    state_c1: np.ndarray
    state_c2: np.ndarray
    mean_power: np.ndarray
    pred_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    succ_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    # 定长表仅用于模型结构检查（测试），推断路径使用分组表。
    succ_idx_fixed: np.ndarray
    pred_idx_fixed: np.ndarray
    on_mask1: np.ndarray
    on_mask2: np.ndarray


def _group_edges(
    edges: list[list[tuple[int, float]]],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """把每个状态的邻居列表按度分组成紧凑数组。"""

    degrees = np.array([len(e) for e in edges], dtype=np.int32)
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for d in np.unique(degrees):
        states = np.where(degrees == d)[0].astype(np.int32)
        idx = np.empty((len(states), int(d)), dtype=np.int32)
        logw = np.empty((len(states), int(d)), dtype=np.float64)
        for row, st in enumerate(states):
            nb = edges[st]
            idx[row] = [j for j, _ in nb]
            logw[row] = [lw for _, lw in nb]
        groups[int(d)] = (states, idx, logw)
    return groups


def _build_model(
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float,
) -> _Model:
    """枚举增强状态并构造前驱/后继转移表。"""

    k1 = max(device1.min_on, device1.min_off)
    k2 = max(device2.min_on, device2.min_off)

    states: list[tuple[int, int, int, int]] = []
    for s1 in (0, 1):
        for s2 in (0, 1):
            for c1 in range(1, k1 + 1):
                for c2 in range(1, k2 + 1):
                    states.append((s1, s2, c1, c2))

    n_states = len(states)
    index = {st: i for i, st in enumerate(states)}
    s1_arr = np.array([st[0] for st in states], dtype=np.int8)
    s2_arr = np.array([st[1] for st in states], dtype=np.int8)
    c1_arr = np.array([st[2] for st in states], dtype=np.int8)
    c2_arr = np.array([st[3] for st in states], dtype=np.int8)

    power1 = np.where(s1_arr == 1, device1.on_power, device1.off_power)
    power2 = np.where(s2_arr == 1, device2.on_power, device2.off_power)
    mean_power = power1 + power2

    p = float(switch_probability)

    successors: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]

    for i, (s1, s2, c1, c2) in enumerate(states):
        can_flip_1 = c1 >= (device1.min_on if s1 == 1 else device1.min_off)
        can_flip_2 = c2 >= (device2.min_on if s2 == 1 else device2.min_off)
        options: list[tuple[bool, bool]] = [(False, False)]
        if can_flip_1:
            options.append((True, False))
        if can_flip_2:
            options.append((False, True))
        if can_flip_1 and can_flip_2:
            options.append((True, True))

        weighted: list[tuple[int, float]] = []
        weight_sum = 0.0
        for flip1, flip2 in options:
            ns1 = 1 - s1 if flip1 else s1
            ns2 = 1 - s2 if flip2 else s2
            nc1 = 1 if flip1 else min(c1 + 1, k1)
            nc2 = 1 if flip2 else min(c2 + 1, k2)
            j = index[(ns1, ns2, nc1, nc2)]
            n_flips = int(flip1) + int(flip2)
            # 两台设备独立翻转概率的乘积，随后在合法选项间归一化，
            # 使每个源状态的条件转移分布之和为 1。
            w = (p ** n_flips) * ((1.0 - p) ** (2 - n_flips))
            weighted.append((j, w))
            weight_sum += w
        for j, w in weighted:
            successors[i].append((j, math.log(w / weight_sum)))

    predecessors: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]
    for i, edges in enumerate(successors):
        for j, lw in edges:
            predecessors[j].append((i, lw))

    succ_groups = _group_edges(successors)
    pred_groups = _group_edges(predecessors)

    # 后继最多 4 个选项；前驱数在非对称最短时长下可超过 8（两台设备
    # 各自的“计数器封顶自环 + 翻转入口”组合，最坏可达 4*(k-min+1) 量级），
    # 因此定长表按实际最大度数分配，推断路径本身使用紧凑分组表。
    max_succ = max(len(e) for e in successors)
    max_pred = max(len(e) for e in predecessors)
    assert max_succ <= 4

    succ_fixed = np.full((n_states, max_succ), -1, dtype=np.int32)
    pred_fixed = np.full((n_states, max_pred), -1, dtype=np.int32)
    for i, edges in enumerate(successors):
        for slot, (j, _) in enumerate(edges):
            succ_fixed[i, slot] = j
    for i, edges in enumerate(predecessors):
        for slot, (j, _) in enumerate(edges):
            pred_fixed[i, slot] = j

    return _Model(
        n_states=n_states,
        k1=k1,
        k2=k2,
        state_s1=s1_arr,
        state_s2=s2_arr,
        state_c1=c1_arr,
        state_c2=c2_arr,
        mean_power=mean_power,
        pred_groups=pred_groups,
        succ_groups=succ_groups,
        succ_idx_fixed=succ_fixed,
        pred_idx_fixed=pred_fixed,
        on_mask1=(s1_arr == 1),
        on_mask2=(s2_arr == 1),
    )


# 编译模型只读、可跨请求共享；缓存本身加锁，推断工作数组全部为
# 请求内局部变量，因此并发请求的中间状态互不污染。
_model_cache: dict[tuple, _Model] = {}
_model_cache_lock = threading.Lock()


def _get_model(
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float,
) -> _Model:
    """按参数键复用编译后的只读模型（线程安全，容量受限）。"""

    key = (
        float(device1.on_power),
        float(device1.off_power),
        int(device1.min_on),
        int(device1.min_off),
        float(device2.on_power),
        float(device2.off_power),
        int(device2.min_on),
        int(device2.min_off),
        float(switch_probability),
    )
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    with _model_cache_lock:
        cached = _model_cache.get(key)
        if cached is None:
            cached = _build_model(device1, device2, switch_probability)
            if len(_model_cache) >= _MODEL_CACHE_MAX:
                # FIFO 淘汰最旧条目，防止异常参数组合无限占内存。
                _model_cache.pop(next(iter(_model_cache)))
            _model_cache[key] = cached
        return cached


def _block_log_emission(
    power: np.ndarray,
    missing: np.ndarray,
    start: int,
    end: int,
    out: np.ndarray,
    model: _Model,
    noise_std: float,
) -> np.ndarray:
    """计算 [start, end) 各行的观测对数似然，写入 out 的前 end-start 行。

    缺测行保持全 0；计算式与全量版本逐元素一致（同样的广播与运算顺序）。
    """

    view = out[: end - start]
    view.fill(0.0)
    block_raw = power[start:end]
    block_missing = missing[start:end]
    observed = ~block_missing
    if observed.any():
        y = block_raw[observed][:, None]  # (O, 1)
        mu = model.mean_power[None, :]  # (1, S)
        z = (y - mu) / noise_std
        view[observed] = -0.5 * z * z - math.log(noise_std) - LOG_SQRT_2PI
    return view


def _grouped_logsumexp(
    values_prev: np.ndarray,
    emit_row: np.ndarray | None,
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """按度分组计算 logsumexp。

    values_prev: 上一时刻（前向）或当前时刻（后向）的 (S,) 对数值。
    emit_row: 后向时需叠加的当前时刻发射对数似然 (S,)；前向传 None。
    返回与每个状态对齐的 (S,) 结果。
    """

    out = np.full(values_prev.shape[0], NEG_INF, dtype=np.float64)
    for d, (states, nb_idx, logw) in groups.items():
        gathered = values_prev[nb_idx] + logw  # (n_d, d)
        if emit_row is not None:
            gathered = gathered + emit_row[nb_idx]
        row_max = gathered.max(axis=-1)
        out[states] = row_max + np.log(
            np.exp(gathered - row_max[:, None]).sum(axis=-1)
        )
    return out


def _checkpointed_forward_backward(
    power: np.ndarray,
    missing: np.ndarray,
    model: _Model,
    noise_std: float,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """分块检查点前向-后向（精确算法，无近似/降采样）。

    返回逐分钟开机后验 on1/on2（长度 T）、缺测掩码（长度 T）与
    序列对数似然。全过程不分配任何完整窗口长度的 (T, S) 矩阵：

    * 前向只保留当前块的 alpha 行，块边界把 alpha 向量存入检查点；
    * 后向以检查点为种子在块内重算 alpha（精确重算 1 遍），同时在
      块大小的 (size, S) 二维数组内反向递推 beta 并归约出逐分钟
      开机边际概率；块结束后这些缓冲即被下一（更早）块复用。
    """

    t_total = power.shape[0]
    n_states = model.n_states
    log_init = -math.log(n_states)  # 初始状态均匀（允许窗口首尾截断）

    n_blocks = math.ceil(t_total / block_size)
    # 检查点：checkpoints[k] 为第 k 块首行 t=k*B 所需的 alpha 种子。
    # 第 0 块用均匀初始分布，无需检查点。
    checkpoints = np.empty((max(n_blocks - 1, 0), n_states), dtype=np.float64)

    # 单块工作缓冲，大小只取决于 block_size 与 n_states，与 T 无关。
    # 多一行用于非末尾块后向时的“前瞻发射”（块首后向一步所需）。
    emit_buf = np.zeros((block_size + 1, n_states), dtype=np.float64)
    alpha_block = np.empty((block_size, n_states), dtype=np.float64)

    # ---- 前向：流式推进，仅在块边界落检查点 ----
    alpha_prev: np.ndarray | None = None
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, t_total)
        size = end - start
        emit = _block_log_emission(
            power, missing, start, end, emit_buf, model, noise_std
        )
        ab = alpha_block[:size]
        if b == 0:
            ab[0] = log_init + emit[0]
        else:
            ab[0] = emit[0] + _grouped_logsumexp(
                alpha_prev, None, model.pred_groups
            )
        for i in range(1, size):
            ab[i] = emit[i] + _grouped_logsumexp(
                ab[i - 1], None, model.pred_groups
            )
        if end < t_total:
            # 下一块首行的种子
            checkpoints[b] = ab[size - 1]
        alpha_prev = ab[size - 1]

    assert alpha_prev is not None
    last = alpha_prev
    row_max = last.max()
    log_likelihood = row_max + np.log(np.exp(last - row_max).sum())
    if not math.isfinite(log_likelihood):
        raise RuntimeError("序列似然为零，参数与观测完全不相容，请检查输入")

    # ---- 后向：逐块回放，块内重算 alpha，beta 只保留单行 ----
    # 块内使用与朴素稠密实现同形的 (size, S) 二维数组（size <= block_size），
    # 不保留任何完整窗口长度的 (T, S) 矩阵；块结束后只留下逐分钟边际概率。
    on1 = np.empty(t_total, dtype=np.float64)
    on2 = np.empty(t_total, dtype=np.float64)
    beta_block = np.zeros((block_size, n_states), dtype=np.float64)
    gamma_block = np.empty((block_size, n_states), dtype=np.float64)
    # 跨块边界 beta 种子的独立存储：不得借用 beta_block 内的槽位，
    # 否则会被下一块的块内递推覆盖。
    beta_seed = np.zeros(n_states, dtype=np.float64)
    mask1 = model.on_mask1
    mask2 = model.on_mask2

    for b in range(n_blocks - 1, -1, -1):
        start = b * block_size
        end = min(start + block_size, t_total)
        size = end - start
        emit = _block_log_emission(
            power, missing, start, end, emit_buf, model, noise_std
        )

        # 非末尾块需要 end 时刻的发射来完成块首的后向一步。
        if end < t_total:
            _block_log_emission(
                power, missing, end, end + 1,
                emit_buf[block_size:block_size + 1], model, noise_std,
            )
            lookahead = emit_buf[block_size]
        else:
            lookahead = None

        # 用检查点种子在块内精确重算 alpha（与前向阶段逐行相同的运算）。
        ab = alpha_block[:size]
        if b == 0:
            ab[0] = log_init + emit[0]
        else:
            ab[0] = emit[0] + _grouped_logsumexp(
                checkpoints[b - 1], None, model.pred_groups
            )
        for i in range(1, size):
            ab[i] = emit[i] + _grouped_logsumexp(
                ab[i - 1], None, model.pred_groups
            )

        # 块内反向递推 beta：二维块与朴素实现逐行同构，beta 只在块内存活。
        bb = beta_block[:size]
        if lookahead is None:
            # 末尾块最后一行：beta(T-1) = 0
            bb[size - 1] = 0.0
        else:
            # 块最后一行的 beta 由上一块（更晚时刻）保存的边界种子与
            # end 时刻发射递推得到。
            bb[size - 1] = _grouped_logsumexp(
                beta_seed, lookahead, model.succ_groups
            )
        for i in range(size - 2, -1, -1):
            bb[i] = _grouped_logsumexp(
                bb[i + 1], emit[i + 1], model.succ_groups
            )

        # 与朴素实现逐元素同形的二维运算：gamma -> 裁剪 -> exp ->
        # 逐行归一化 -> 开机切片 axis=1 求和；归约后块矩阵即被复用。
        # 注意保持单条 (ab + bb - ll) 表达式，与稠密版舍入路径一致。
        gb = gamma_block[:size]
        np.subtract(ab + bb, log_likelihood, out=gb)
        np.clip(gb, NEG_INF, 0.0, out=gb)
        post_block = np.exp(gb)
        post_block = post_block / post_block.sum(axis=1, keepdims=True)
        if size == 1:
            # NumPy 对单行 (1, k) 与多行 (n, k) 的 axis=1 归约走不同内核，
            # 末位舍入可能不同；这里补齐一行再归约，与稠密路径保持同构。
            pad = np.broadcast_to(post_block[0], (2, n_states))
            on1[start:end] = pad[:, mask1].sum(axis=1)[:1]
            on2[start:end] = pad[:, mask2].sum(axis=1)[:1]
        else:
            on1[start:end] = post_block[:, mask1].sum(axis=1)
            on2[start:end] = post_block[:, mask2].sum(axis=1)
        # 保存本块块首 beta 作为下一（更早）块的边界种子。
        beta_seed = bb[0].copy()

    np.clip(on1, 0.0, 1.0, out=on1)
    np.clip(on2, 0.0, 1.0, out=on2)
    return on1, on2, missing, log_likelihood


def infer_posterior(
    power: list[float | None],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float = 0.1,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """运行完整推断，返回两台设备逐分钟的开机后验概率与区间。

    参数
    ----
    power: 逐分钟总功率，元素为 None 表示该分钟缺测。
    noise_std: 电表读数噪声标准差（正值）。
    device1, device2: 两台设备的参数。
    switch_probability: 单台设备单分钟发生状态翻转的先验概率。
    block_size: 检查点分块大小（分钟），仅影响内存/重算的平衡点，
        不影响数值结果。

    返回
    ----
    字典，包含 posterior_on（2 x T 后验概率）、missing、log_likelihood 等。
    """

    if not power:
        raise ValueError("power 序列不能为空")
    t_total = len(power)
    if t_total > 1440:
        raise ValueError("观测序列最长为一天（1440 分钟）")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size 必须为正整数")

    raw = np.array(
        [np.nan if v is None else float(v) for v in power], dtype=np.float64
    )
    missing = np.isnan(raw)
    model = _get_model(device1, device2, switch_probability)
    on1, on2, missing, log_likelihood = _checkpointed_forward_backward(
        raw, missing, model, float(noise_std), block_size
    )

    return {
        "posterior_on": [on1.tolist(), on2.tolist()],
        "missing": missing.tolist(),
        "log_likelihood": log_likelihood,
        "n_states": model.n_states,
        "window_minutes": t_total,
    }


def label_intervals(
    posterior_on: list[float],
    missing: list[bool],
    confidence_threshold: float,
) -> list[dict]:
    """把逐分钟后验概率压缩为 running / stopped / uncertain 区间。

    * p_on >= 阈值 -> running；p_on <= 1-阈值 -> stopped；其余 uncertain。
    * 相邻同标签分钟合并为半开区间 [start_minute, end_minute)。
    * 触及窗口首/尾的区间在对应边缘标记 truncated（状态被窗口截断）。
    * 区间置信度取区间内各分钟置信度的最小值（保守口径）。

    最短时长约束已在 HSMM 转移层强制，标签层不重复截断。
    """

    if not 0.5 < confidence_threshold < 1.0:
        raise ValueError("confidence_threshold 必须位于 (0.5, 1.0) 之间")

    n = len(posterior_on)
    labels: list[int] = []
    confidences: list[float] = []
    for p in posterior_on:
        if p >= confidence_threshold:
            labels.append(1)
            confidences.append(p)
        elif p <= 1.0 - confidence_threshold:
            labels.append(0)
            confidences.append(1.0 - p)
        else:
            labels.append(-1)
            confidences.append(max(p, 1.0 - p))

    intervals: list[dict] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            label = labels[start]
            state = {1: "running", 0: "stopped", -1: "uncertain"}[label]
            end = i
            truncated_edges: list[str] = []
            if start == 0:
                truncated_edges.append("start")
            if end == n:
                truncated_edges.append("end")
            missing_count = sum(1 for k in range(start, end) if missing[k])
            intervals.append(
                {
                    "state": state,
                    "start_minute": start,
                    "end_minute": end,
                    "duration_minutes": end - start,
                    "missing_minutes": missing_count,
                    "confidence": round(min(confidences[start:end]), 6),
                    "truncated_edges": truncated_edges,
                    "truncated": bool(truncated_edges),
                }
            )
            start = i
    return intervals
