"""全天（1440 分钟、最短时长均 30、S=3600）内存与耗时实测。

两种实现分别在**独立子进程**中运行，用资源占用峰值 (ru_maxrss) 测量
峰值内存，避免相互干扰；同一组输入下逐值比对两份输出。

用法：
  .venv/bin/python -I benchmarks/run_benchmark.py            # 内存+耗时+对照
  .venv/bin/python -I benchmarks/run_benchmark.py --sweep    # 额外扫块大小
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.hsmm import DeviceParams, infer_posterior  # noqa: E402

D1 = DeviceParams(on_power=10.0, off_power=0.5, min_on=30, min_off=30)
D2 = DeviceParams(on_power=20.0, off_power=1.0, min_on=30, min_off=30)
NOISE_STD = 1.0
TOTAL = 1440
TIMING_REPEATS = 5


def build_input(seed: int = 8) -> list[float | None]:
    """构造带真实开关节律与随机缺测的全天序列。"""

    rng = np.random.default_rng(seed)
    s1 = [0] * 200 + [1] * 300 + [0] * 240 + [1] * 200 + [0] * 500
    s2 = [1] * 180 + [0] * 260 + [1] * 320 + [0] * 680
    power: list[float | None] = []
    for t in range(TOTAL):
        if t % 97 == 0:  # 15 个缺测分钟
            power.append(None)
            continue
        mu = (D1.on_power if s1[t] else D1.off_power) + (
            D2.on_power if s2[t] else D2.off_power
        )
        power.append(float(mu + rng.normal(0.0, NOISE_STD)))
    return power


def _fingerprint(out: dict) -> dict:
    p1 = np.asarray(out["posterior_on"][0])
    p2 = np.asarray(out["posterior_on"][1])
    return {
        "log_likelihood": out["log_likelihood"],
        "sum_on1": float(p1.sum()),
        "sum_on2": float(p2.sum()),
        "first8_on1": p1[:8].tolist(),
        "last8_on2": p2[-8:].tolist(),
    }


def run_chunked(block_size: int, measure_mem: bool) -> dict:
    power = build_input()
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    times = []
    out = None
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        out = infer_posterior(
            power, NOISE_STD, D1, D2, block_size=block_size
        )
        times.append(time.perf_counter() - t0)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "mode": f"checkpointed(B={block_size})",
        "n_states": out["n_states"],
        "median_time_s": float(np.median(times)),
        "min_time_s": float(min(times)),
        "fingerprint": _fingerprint(out),
    }
    if measure_mem:
        result.update(
            peak_rss_mb=peak_rss / 1024.0,
            delta_rss_mb=(peak_rss - base_rss) / 1024.0,
        )
    return result


def run_dense(measure_mem: bool) -> dict:
    from benchmarks.dense_baseline import dense_infer

    power = build_input()
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    times = []
    out = None
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        out = dense_infer(power, NOISE_STD, D1, D2)
        times.append(time.perf_counter() - t0)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "mode": "dense_baseline",
        "n_states": out["n_states"],
        "median_time_s": float(np.median(times)),
        "min_time_s": float(min(times)),
        "fingerprint": _fingerprint(out),
    }
    if measure_mem:
        result.update(
            peak_rss_mb=peak_rss / 1024.0,
            delta_rss_mb=(peak_rss - base_rss) / 1024.0,
        )
    return result


def elementwise_compare() -> dict:
    """同进程内逐值比对稠密基线与检查点实现（含多个块大小）。"""

    power = build_input()
    from benchmarks.dense_baseline import dense_infer

    ref = dense_infer(power, NOISE_STD, D1, D2)
    report = {}
    for bs in (1, 33, 64, 128, 256, 1440):
        got = infer_posterior(power, NOISE_STD, D1, D2, block_size=bs)
        max_abs = 0.0
        for dev in range(2):
            diff = np.abs(
                np.asarray(got["posterior_on"][dev])
                - np.asarray(ref["posterior_on"][dev])
            )
            max_abs = max(max_abs, float(diff.max()))
        ll_diff = abs(got["log_likelihood"] - ref["log_likelihood"])
        report[f"B={bs}"] = {
            "max_abs_posterior_diff": max_abs,
            "abs_log_likelihood_diff": ll_diff,
            "missing_equal": got["missing"] == ref["missing"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="扫描多种块大小")
    parser.add_argument("--worker", choices=("chunked", "dense"), default=None)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    if args.worker == "chunked":
        print(json.dumps(run_chunked(args.block_size, measure_mem=True)))
        return
    if args.worker == "dense":
        print(json.dumps(run_dense(measure_mem=True)))
        return

    # 父进程：分别拉起独立子进程测峰值内存
    import subprocess

    def spawn(worker: str, block_size: int | None = None) -> dict:
        cmd = [
            sys.executable, "-I", str(Path(__file__).resolve()),
            "--worker", worker,
        ]
        if block_size is not None:
            cmd += ["--block-size", str(block_size)]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(proc.stdout.strip())

    dense_res = spawn("dense")
    chunk_res = spawn("chunked", 128)

    print("=" * 72)
    print("全天 1440 分钟 / 最短时长均 30 / 增强状态 S=3600 / 含 15 个缺测")
    print("=" * 72)
    for r in (dense_res, chunk_res):
        print(
            f"\n[{r['mode']}]  S={r['n_states']}\n"
            f"  耗时(5 次中位/最快): {r['median_time_s']:.3f}s / {r['min_time_s']:.3f}s\n"
            f"  进程峰值 RSS: {r['peak_rss_mb']:.1f} MB"
            f"（推断相对增量 {r['delta_rss_mb']:.1f} MB）"
        )

    print("\n输出指纹（两实现必须一致）:")
    print(json.dumps(dense_res["fingerprint"], indent=2, ensure_ascii=False))
    fp_ok = (
        chunk_res["fingerprint"]["log_likelihood"]
        == dense_res["fingerprint"]["log_likelihood"]
        and chunk_res["fingerprint"]["sum_on1"]
        == dense_res["fingerprint"]["sum_on1"]
        and chunk_res["fingerprint"]["sum_on2"]
        == dense_res["fingerprint"]["sum_on2"]
        and chunk_res["fingerprint"]["first8_on1"]
        == dense_res["fingerprint"]["first8_on1"]
        and chunk_res["fingerprint"]["last8_on2"]
        == dense_res["fingerprint"]["last8_on2"]
    )
    assert fp_ok, "指纹不一致！"
    print("-> 检查点实现与稠密基线指纹完全相同 ✓")

    print("\n逐值对照（稠密基线 vs 检查点，多种块大小）:")
    cmp_report = elementwise_compare()
    for name, info in cmp_report.items():
        print(
            f"  {name:>8}: max|Δposterior|={info['max_abs_posterior_diff']:.3e}"
            f"  |Δll|={info['abs_log_likelihood_diff']:.3e}"
            f"  missing一致={info['missing_equal']}"
        )

    # 理论工作集（不含解释器/模型/输出），单位 MB
    t_total, n_states, bs = 1440, 3600, 128
    n_blocks = math.ceil(t_total / bs)
    matrices = {
        "alpha/beta/emission 单块工作缓冲 ((B+1+B)*S*8B)": (2 * bs + 1) * n_states * 8 / 1e6,
        f"alpha 检查点 (({n_blocks}-1)*S*8B)": (n_blocks - 1) * n_states * 8 / 1e6,
        "旧实现 alpha+beta+emit (3*T*S*8B)": 3 * t_total * n_states * 8 / 1e6,
    }
    print("\n理论工作集估算（浮点缓冲，不含解释器/模型/最终输出）:")
    for name, mb in matrices.items():
        print(f"  {name}: {mb:.2f} MB")

    saved = dense_res["peak_rss_mb"] - chunk_res["peak_rss_mb"]
    print(
        f"\n峰值 RSS 下降: {dense_res['peak_rss_mb']:.1f} -> "
        f"{chunk_res['peak_rss_mb']:.1f} MB（减少 {saved:.1f} MB，"
        f"{saved / dense_res['peak_rss_mb'] * 100:.0f}%）"
    )

    if args.sweep:
        print("\n块大小扫描（同进程耗时，3 次中位）:")
        for bs in (16, 32, 64, 128, 256, 512, 1440):
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                infer_posterior(build_input(), NOISE_STD, D1, D2, block_size=bs)
                times.append(time.perf_counter() - t0)
            nblk = math.ceil(t_total / bs)
            print(
                f"  B={bs:>4}: {np.median(times):.3f}s  "
                f"块数={nblk:>3}  检查点内存={ (nblk-1)*n_states*8/1e6:.2f}MB"
            )


if __name__ == "__main__":
    main()
