"""优化前的稠密 (T, S) 前向-后向实现（基准对照，不参与服务路径）。

本文件逐行保留重构前 app/hsmm.py 的推断逻辑：完整分配 alpha、beta、
emission 三个 (T, S) 矩阵。仅用于 benchmarks/run_benchmark.py 做
内存/耗时对比与位级等价核验，生产代码不导入本模块。
模型结构直接复用当前 app.hsmm 的编译结果（只读转移表），
以保证对比的唯一变量是“分块检查点”这一内存策略。
"""

from __future__ import annotations

import math

import numpy as np

from app.hsmm import (
    NEG_INF,
    LOG_SQRT_2PI,
    DeviceParams,
    _get_model,
    _grouped_logsumexp,
)


def dense_infer(
    power: list[float | None],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float = 0.1,
) -> dict:
    """重构前的稠密实现：峰值内存 O(T*S)。"""

    t_total = len(power)
    raw = np.array(
        [np.nan if v is None else float(v) for v in power], dtype=np.float64
    )
    missing = np.isnan(raw)
    model = _get_model(device1, device2, switch_probability)

    # 完整 (T, S) 发射矩阵
    emit = np.zeros((t_total, model.n_states), dtype=np.float64)
    observed = ~missing
    if observed.any():
        y = raw[observed][:, None]
        mu = model.mean_power[None, :]
        z = (y - mu) / noise_std
        emit[observed] = -0.5 * z * z - math.log(noise_std) - LOG_SQRT_2PI

    n_states = model.n_states
    log_init = -math.log(n_states)

    # 完整 (T, S) alpha
    alpha = np.empty((t_total, n_states), dtype=np.float64)
    alpha[0] = log_init + emit[0]
    for t in range(1, t_total):
        alpha[t] = emit[t] + _grouped_logsumexp(
            alpha[t - 1], None, model.pred_groups
        )

    last = alpha[t_total - 1]
    row_max = last.max()
    log_likelihood = row_max + np.log(np.exp(last - row_max).sum())

    # 完整 (T, S) beta
    beta = np.zeros((t_total, n_states), dtype=np.float64)
    for t in range(t_total - 1, 0, -1):
        beta[t - 1] = _grouped_logsumexp(
            beta[t], emit[t], model.succ_groups
        )

    # 完整 (T, S) gamma/posterior
    gamma = alpha + beta - log_likelihood
    gamma = np.clip(gamma, NEG_INF, 0.0)
    posterior = np.exp(gamma)
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    on1 = np.clip(posterior[:, model.state_s1 == 1].sum(axis=1), 0.0, 1.0)
    on2 = np.clip(posterior[:, model.state_s2 == 1].sum(axis=1), 0.0, 1.0)

    return {
        "posterior_on": [on1.tolist(), on2.tolist()],
        "missing": missing.tolist(),
        "log_likelihood": log_likelihood,
        "n_states": model.n_states,
        "window_minutes": t_total,
    }
