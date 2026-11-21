"""独立参考实现：小规模稠密对数空间前向-后向，仅用于逐值对照测试。

本模块刻意不导入 app.hsmm 的任何推断例程，状态枚举、转移构造、
logsumexp 与前向-后向全部独立重写（使用显式 Python 循环 + 朴素
稠密矩阵），以便作为“第二份实现”交叉验证分块检查点版本的正确性。
仅适合测试中的小规模输入（T <= 数百、状态数 <= 数百）。
"""

from __future__ import annotations

import math

import numpy as np

from app.hsmm import DeviceParams


def _logsumexp(xs: list[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(x - m) for x in finite))


def reference_infer(
    power: list[float | None],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float,
) -> dict:
    """稠密 (T, S) 前向-后向参考实现，返回与 infer_posterior 同构的字典。"""

    k1 = max(device1.min_on, device1.min_off)
    k2 = max(device2.min_on, device2.min_off)

    # 独立枚举增强状态（顺序与生产实现独立，不共享索引约定）。
    states: list[tuple[int, int, int, int]] = []
    for c2 in range(1, k2 + 1):
        for c1 in range(1, k1 + 1):
            for s2 in (0, 1):
                for s1 in (0, 1):
                    states.append((s1, s2, c1, c2))
    n_states = len(states)
    index = {st: i for i, st in enumerate(states)}

    # 独立构造归一化转移概率（普通字典邻接表，不用分组紧凑数组）。
    trans_log: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_states)}
    p = float(switch_probability)
    for i, (s1, s2, c1, c2) in enumerate(states):
        can1 = c1 >= (device1.min_on if s1 == 1 else device1.min_off)
        can2 = c2 >= (device2.min_on if s2 == 1 else device2.min_off)
        options = [(False, False)]
        if can1:
            options.append((True, False))
        if can2:
            options.append((False, True))
        if can1 and can2:
            options.append((True, True))
        raw_edges: list[tuple[int, float]] = []
        wsum = 0.0
        for f1, f2 in options:
            ns1 = 1 - s1 if f1 else s1
            ns2 = 1 - s2 if f2 else s2
            nc1 = 1 if f1 else min(c1 + 1, k1)
            nc2 = 1 if f2 else min(c2 + 1, k2)
            j = index[(ns1, ns2, nc1, nc2)]
            nf = int(f1) + int(f2)
            w = (p ** nf) * ((1.0 - p) ** (2 - nf))
            raw_edges.append((j, w))
            wsum += w
        for j, w in raw_edges:
            trans_log[i].append((j, math.log(w / wsum)))

    # 前驱表独立构造。
    preds: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_states)}
    for i, edges in trans_log.items():
        for j, lw in edges:
            preds[j].append((i, lw))

    t_total = len(power)
    means = np.array(
        [
            (device1.on_power if st[0] == 1 else device1.off_power)
            + (device2.on_power if st[1] == 1 else device2.off_power)
            for st in states
        ],
        dtype=np.float64,
    )

    # 独立计算发射对数似然（逐标量循环，不与生产实现共享向量化代码）。
    log_emit = [[0.0] * n_states for _ in range(t_total)]
    missing = [v is None for v in power]
    log_norm = math.log(noise_std) + 0.5 * math.log(2.0 * math.pi)
    for t, v in enumerate(power):
        if v is None:
            continue
        for i in range(n_states):
            z = (float(v) - float(means[i])) / noise_std
            log_emit[t][i] = -0.5 * z * z - log_norm

    # 稠密前向：完整保存 alpha (T, S)。
    log_init = -math.log(n_states)
    alpha = [[-math.inf] * n_states for _ in range(t_total)]
    for i in range(n_states):
        alpha[0][i] = log_init + log_emit[0][i]
    for t in range(1, t_total):
        for j in range(n_states):
            terms = [
                alpha[t - 1][i] + lw + log_emit[t][j]
                for i, lw in preds[j]
            ]
            alpha[t][j] = _logsumexp(terms)

    log_likelihood = _logsumexp(alpha[t_total - 1])

    # 稠密后向：完整保存 beta (T, S)。
    beta = [[-math.inf] * n_states for _ in range(t_total)]
    for i in range(n_states):
        beta[t_total - 1][i] = 0.0
    for t in range(t_total - 1, 0, -1):
        for i in range(n_states):
            terms = [
                lw + log_emit[t][j] + beta[t][j]
                for j, lw in trans_log[i]
            ]
            beta[t - 1][i] = _logsumexp(terms)

    # gamma 与逐设备开机边际。
    on1 = [0.0] * t_total
    on2 = [0.0] * t_total
    for t in range(t_total):
        posterior = [
            math.exp(max(alpha[t][i] + beta[t][i] - log_likelihood, -1e300))
            for i in range(n_states)
        ]
        z = sum(posterior)
        posterior = [q / z for q in posterior]
        for i, (s1, s2, _c1, _c2) in enumerate(states):
            if s1 == 1:
                on1[t] += posterior[i]
            if s2 == 1:
                on2[t] += posterior[i]

    return {
        "posterior_on": [on1, on2],
        "missing": missing,
        "log_likelihood": log_likelihood,
        "n_states": n_states,
        "window_minutes": t_total,
    }
