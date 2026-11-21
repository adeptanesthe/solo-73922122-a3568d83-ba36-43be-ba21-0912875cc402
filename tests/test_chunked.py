"""分块检查点前向-后向的逐值对照测试。

对照对象是 tests/reference_impl.py 中完全独立的稠密参考实现：
状态枚举顺序、转移构造、logsumexp、前向-后向均独立重写。
覆盖：不同窗口长度（含 T=1）、随机缺测、非零待机功率、不同翻转先验、
不同块大小（含不能整除窗口的块）、全天最坏规模的块大小不变性，
以及并发请求的中间状态隔离。
"""

from __future__ import annotations

import concurrent.futures
import math
import unittest

import numpy as np

from app.hsmm import DEFAULT_BLOCK_SIZE, DeviceParams, infer_posterior
from tests.reference_impl import reference_infer


def _make_power(s1, s2, d1, d2, noise_std, seed, missing_prob=0.0):
    rng = np.random.default_rng(seed)
    out: list[float | None] = []
    for t in range(len(s1)):
        if rng.random() < missing_prob:
            out.append(None)
            continue
        mu = (d1.on_power if s1[t] else d1.off_power) + (
            d2.on_power if s2[t] else d2.off_power
        )
        out.append(float(mu + rng.normal(0.0, noise_std)))
    return out


def _random_schedule(t_total, min_run, rng):
    s = []
    cur = int(rng.integers(0, 2))
    while len(s) < t_total:
        run = int(rng.integers(min_run, min_run + 8))
        s.extend([cur] * min(run, t_total - len(s)))
        cur = 1 - cur
    return s


class TestAgainstIndependentReference(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def _assert_matches_reference(
        self, power, d1, d2, noise_std, switch_probability, tol=1e-10
    ):
        ref = reference_infer(power, noise_std, d1, d2, switch_probability)
        for block_size in (1, 2, 3, 7, 16, 31, 64, 128, 10_000):
            with self.subTest(block_size=block_size, t=len(power)):
                got = infer_posterior(
                    power,
                    noise_std,
                    d1,
                    d2,
                    switch_probability=switch_probability,
                    block_size=block_size,
                )
                self.assertEqual(got["n_states"], ref["n_states"])
                self.assertEqual(got["window_minutes"], ref["window_minutes"])
                self.assertEqual(got["missing"], ref["missing"])
                self.assertAlmostEqual(
                    got["log_likelihood"],
                    ref["log_likelihood"],
                    delta=max(1e-9, abs(ref["log_likelihood"]) * 1e-12),
                )
                for dev in range(2):
                    np.testing.assert_allclose(
                        np.asarray(got["posterior_on"][dev]),
                        np.asarray(ref["posterior_on"][dev]),
                        rtol=tol,
                        atol=tol,
                        err_msg=f"设备{dev+1} 后验逐值不一致 (B={block_size})",
                    )
                    # 概率合法性
                    p = np.asarray(got["posterior_on"][dev])
                    self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))

    def test_tiny_window_lengths(self):
        d1 = DeviceParams(3.0, 0.2, 2, 2)
        d2 = DeviceParams(5.0, 0.5, 3, 2)
        # T=1：只有初始行，不发生任何转移，后向恒为 0
        for t_total in (1, 2, 3, 5):
            power = _make_power(
                [0] * t_total, [1] * t_total, d1, d2, noise_std=0.4, seed=100 + t_total
            )
            self._assert_matches_reference(power, d1, d2, 0.4, 0.1)

    def test_missing_patterns(self):
        d1 = DeviceParams(10.0, 0.0, 4, 4)
        d2 = DeviceParams(20.0, 1.0, 3, 5)
        t_total = 60
        s1 = [0] * 20 + [1] * 20 + [0] * 20
        s2 = [1] * 30 + [0] * 30
        # 前缀缺测、内部缺测、全缺测、随机缺测
        cases = {
            "prefix": [None] * 8 + _make_power(s1, s2, d1, d2, 0.6, 1)[8:],
            "inner": (
                _make_power(s1, s2, d1, d2, 0.6, 2)[:25]
                + [None] * 10
                + _make_power(s1, s2, d1, d2, 0.6, 2)[35:]
            ),
            "all_missing": [None] * t_total,
            "random": _make_power(s1, s2, d1, d2, 0.6, 3, missing_prob=0.25),
        }
        for name, power in cases.items():
            with self.subTest(case=name):
                self._assert_matches_reference(power, d1, d2, 0.6, 0.1)

    def test_nonzero_standby_and_varied_priors(self):
        d1 = DeviceParams(7.5, 1.2, 5, 6)
        d2 = DeviceParams(13.0, 2.1, 4, 7)
        for sp in (0.01, 0.25, 0.7):
            s1 = _random_schedule(80, 6, np.random.default_rng(7))
            s2 = _random_schedule(80, 5, np.random.default_rng(8))
            power = _make_power(s1, s2, d1, d2, noise_std=1.5, seed=11, missing_prob=0.1)
            with self.subTest(switch_probability=sp):
                self._assert_matches_reference(power, d1, d2, 1.5, sp)

    def test_min_duration_equals_one(self):
        d1 = DeviceParams(4.0, 0.0, 1, 1)
        d2 = DeviceParams(9.0, 0.0, 1, 1)
        s1 = _random_schedule(50, 1, np.random.default_rng(3))
        s2 = _random_schedule(50, 1, np.random.default_rng(4))
        power = _make_power(s1, s2, d1, d2, noise_std=0.7, seed=5)
        self._assert_matches_reference(power, d1, d2, 0.7, 0.3)

    def test_randomized_small_cases(self):
        rng = np.random.default_rng(123)
        for case in range(8):
            k = int(rng.integers(1, 6))
            d1 = DeviceParams(
                float(rng.uniform(5, 15)),
                float(rng.uniform(0, 2)),
                k,
                int(rng.integers(1, 6)),
            )
            d2 = DeviceParams(
                float(rng.uniform(15, 30)),
                float(rng.uniform(0, 2)),
                int(rng.integers(1, 6)),
                k,
            )
            t_total = int(rng.integers(1, 90))
            s1 = _random_schedule(t_total, 1, rng)
            s2 = _random_schedule(t_total, 1, rng)
            noise = float(rng.uniform(0.3, 4.0))
            power = _make_power(
                s1, s2, d1, d2, noise_std=noise, seed=case,
                missing_prob=float(rng.uniform(0.0, 0.3)),
            )
            with self.subTest(case=case):
                self._assert_matches_reference(
                    power, d1, d2, noise, float(rng.uniform(0.02, 0.5))
                )


class TestBlockSizeInvarianceFullDay(unittest.TestCase):
    def test_full_day_worst_case_all_block_sizes(self):
        # 最坏规模：最短时长均 30（3600 状态）、全天 1440 分钟、随机缺测。
        d1 = DeviceParams(10.0, 0.5, 30, 30)
        d2 = DeviceParams(20.0, 1.0, 30, 30)
        rng = np.random.default_rng(99)
        s1 = _random_schedule(1440, 30, rng)
        s2 = _random_schedule(1440, 30, rng)
        power = _make_power(
            s1, s2, d1, d2, noise_std=1.2, seed=99, missing_prob=0.05
        )
        ref_block = infer_posterior(power, 1.2, d1, d2, block_size=128)
        for block_size in (1, 5, 33, 64, 127, 128, 129, 256, 512, 1440, 2000):
            with self.subTest(block_size=block_size):
                got = infer_posterior(
                    power, 1.2, d1, d2, block_size=block_size
                )
                self.assertAlmostEqual(
                    got["log_likelihood"], ref_block["log_likelihood"], places=12
                )
                for dev in range(2):
                    np.testing.assert_allclose(
                        got["posterior_on"][dev],
                        ref_block["posterior_on"][dev],
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_invalid_block_size_rejected(self):
        d1 = DeviceParams(10.0, 0.0, 2, 2)
        d2 = DeviceParams(20.0, 0.0, 2, 2)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                infer_posterior([1.0, 2.0], 1.0, d1, d2, block_size=bad)


class TestConcurrentIsolation(unittest.TestCase):
    def test_concurrent_requests_do_not_cross_contaminate(self):
        d1 = DeviceParams(10.0, 0.0, 5, 5)
        d2 = DeviceParams(20.0, 0.0, 3, 3)
        rng = np.random.default_rng(77)

        jobs = []
        for j in range(12):
            t_total = 120 + j * 37
            s1 = _random_schedule(t_total, 5, np.random.default_rng(1000 + j))
            s2 = _random_schedule(t_total, 4, np.random.default_rng(2000 + j))
            noise = 0.5 + 0.1 * j
            power = _make_power(
                s1, s2, d1, d2, noise_std=noise, seed=j,
                missing_prob=0.1 if j % 2 else 0.0,
            )
            jobs.append((power, noise, j))

        def run(job):
            power, noise, j = job
            # 刻意混用不同块大小，确认检查点缓冲均为请求内局部对象
            bs = (16, 64, 128, 333)[j % 4]
            return infer_posterior(power, noise, d1, d2, block_size=bs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, jobs))

        # 单线程重跑并逐值比对；并发结果必须与隔离执行完全一致。
        for job, concurrent_out in zip(jobs, results):
            power, noise, j = job
            bs = (16, 64, 128, 333)[j % 4]
            serial_out = infer_posterior(power, noise, d1, d2, block_size=bs)
            self.assertEqual(
                concurrent_out["window_minutes"], serial_out["window_minutes"]
            )
            self.assertAlmostEqual(
                concurrent_out["log_likelihood"],
                serial_out["log_likelihood"],
                places=12,
            )
            for dev in range(2):
                np.testing.assert_array_equal(
                    np.asarray(concurrent_out["posterior_on"][dev]),
                    np.asarray(serial_out["posterior_on"][dev]),
                )
            self.assertEqual(concurrent_out["missing"], serial_out["missing"])

        # 不同请求的窗口长度确实不同，避免“假并发”通过
        lengths = {r["window_minutes"] for r in results}
        self.assertGreater(len(lengths), 1)


if __name__ == "__main__":
    unittest.main()
