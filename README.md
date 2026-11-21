# 双设备开停时段推断服务

食堂值班员只有一块总电表，需要核对两台设备各自的开停时段。本服务接收
**至多一天（1–1440 分钟）的逐分钟总功率**，用显式时长隐半马尔可夫模型
（HSMM）联合推断两台设备的运行/停机/待核实时段。

## 模型与判定口径

- **隐状态**：两台设备状态的笛卡尔积 `(s1, s2)`，各自带“当前状态已持续
  分钟数”计数器；计数器到上限后封顶，任意长度的开/停段都能表达。
- **最短时长约束（1–30 分钟）**：设备只有在当前开/停状态持续满最短分钟数
  后才允许翻转，约束直接编码在状态转移表中。
- **观测模型**：总功率 = 两设备当前功率之和，叠加高斯噪声 `N(0, noise_std²)`；
  开机/停机功率分别标定，支持非零待机功率。
- **缺测分钟**：`null` 不产生观测似然，但转移照常推进，**缺测分钟仍计入
  开/停时长**，并在输出区间的 `missing_minutes` 中单独统计。
- **窗口首尾截断**：初始分布对所有增强状态均匀，不要求首个状态已持续满
  最短时长；触及窗口首/尾的区间用 `truncated` / `truncated_edges` 标记。
- **置信度判定**：由前向-后向算法得到逐分钟开机后验概率 `p`：
  - `p >= confidence_threshold` → `running`
  - `p <= 1 - confidence_threshold` → `stopped`
  - 其余 → `uncertain`（待核实）
  - 阈值必须 **大于 0.5**；区间置信度取区间内逐分钟置信度的最小值（保守口径）。
- 当两台设备功率组合本身不可区分（结构性歧义）时，相关分钟会自然落入
  `uncertain`，不会给出武断结论。

## 目录结构

```
app/
  hsmm.py      # HSMM 状态空间、转移表、分块检查点前向-后向、区间标注
  schemas.py   # Pydantic 请求/响应模型与输入校验
  main.py      # FastAPI 路由（/health、/infer、/docs）
tests/         # unittest 自测（模型性质 + 接口校验 + 独立参考逐值对照 + 并发隔离）
benchmarks/    # 稠密基线与全天内存/耗时实测脚本
examples/      # 请求样例与调用脚本
Dockerfile
docker-compose.yml
```

## 内存实现：分块检查点前向-后向（精确，无近似）

朴素前向-后向需要保存完整的 `(T, S)` alpha / beta / emission 矩阵
（全天 `T=1440`、最坏规模 `S=3600` 时，仅这三个 float64 矩阵就约
124 MB，进程峰值 RSS 实测约 333 MB），峰值内存随窗口长度线性增长。

本服务把时间轴切成固定大小的块（默认 **128 分钟**，
`app.hsmm.DEFAULT_BLOCK_SIZE`）：

1. **前向流式推进**：只保留当前块的 alpha 行，块边界把当时的 alpha
   向量（`S` 个 float64）存入检查点，检查点数量为 `ceil(T/B)-1`；
2. **后向逐块回放**：以边界检查点为种子，在块内**精确重算**一遍该块
   alpha（重算量恰好 1 倍，不做任何近似、降采样或阈值截断），同时反向
   递推 beta；块内用与朴素实现同形的 `(size, S)` 二维数组计算
   gamma / 后验 / 开机边际，块结束后矩阵缓冲即被复用，不保留任何完整
   窗口长度的 `(T, S)` 矩阵。

因此峰值工作内存为 `O(B·S + (T/B)·S)`，**由固定块大小 B 和状态数 S
决定，与窗口长度 T 不再是线性关系**。浮点运算顺序与朴素实现逐行对齐，
多种块大小（含 `B=1`、不能整除窗口的块）下与重构前稠密实现
**位级一致（max|Δ| = 0）**，缺测、首尾截断、逐分钟后验、区间标注与
`log_likelihood` 的语义保持不变，HTTP 响应格式与字段完全不变。

编译后的转移模型为只读结构，按参数缓存并在请求间共享（容量受限、
带锁）；所有推断工作数组都是请求内局部对象，并发请求的中间状态
互不污染（见 `tests/test_chunked.py::TestConcurrentIsolation`）。

### 全天实测（1440 分钟、最短时长均 30、S=3600、含缺测）

```bash
.venv/bin/python -I benchmarks/run_benchmark.py --sweep
```

| 实现 | 耗时（5 次中位） | 进程峰值 RSS | 推断相对增量 |
| --- | --- | --- | --- |
| 重构前稠密 `(T,S)` 矩阵 | 约 1.3 s | 约 333 MB | 约 299 MB |
| 分块检查点 `B=128` | 约 1.6 s | **约 65 MB** | 约 30 MB |

峰值 RSS 下降约 **81%**；理论浮点工作缓冲（不含解释器/模型/最终输出）
从约 124 MB 降到约 8 MB（单块缓冲 7.4 MB + 检查点 0.32 MB）。
块大小只影响内存/重算平衡点，不改变数值结果；`B` 越小检查点越多，
`B=1440` 时退化为单块（无检查点）。

### 独立参考实现的逐值对照

`tests/reference_impl.py` 是一份**完全独立**的小规模参考实现：状态枚举
顺序、转移构造、logsumexp、稠密前向-后向全部独立重写（显式 Python
循环，不导入生产推断代码）。`tests/test_chunked.py` 在不同窗口长度
（含 `T=1`）、随机/前缀/内部/全缺测、非零待机功率、不同翻转先验与
多种块大小下逐值对照，容差 1e-10；另有与重构前稠密基线的位级对照
（差异严格为 0）。

## 用 Docker Compose 启动

```bash
docker compose up --build
```

服务监听 `http://127.0.0.1:8000`：

- 健康检查：`GET /health`
- 交互式文档：`GET /docs`（Swagger UI）
- OpenAPI Schema：`GET /openapi.json`
- 推断接口：`POST /infer`

停止：`docker compose down`

## 调用样例

### curl

```bash
curl -s http://127.0.0.1:8000/infer \
  -H 'Content-Type: application/json' \
  -d @examples/request.json | python -m json.tool
```

### Python

```bash
python examples/call_example.py
```

### 请求字段

| 字段 | 说明 |
| --- | --- |
| `power` | 逐分钟总功率数组，长度 1–1440；`null` 表示该分钟缺测 |
| `noise_std` | 读数噪声标准差，必须 > 0 |
| `device1` / `device2` | `name`、`on_power`、`off_power`、`min_on_minutes`、`min_off_minutes`（最短时长 1–30） |
| `confidence_threshold` | 置信度阈值，默认 0.95，必须 ∈ (0.5, 1.0) |
| `switch_probability` | 单设备单分钟翻转的先验概率，默认 0.1，∈ (0, 1) |

### 响应摘要

每台设备返回：

- `posterior_on`：逐分钟开机后验概率；
- `intervals`：合并后的区间，字段包括 `state`（running/stopped/uncertain）、
  `[start_minute, end_minute)` 半开区间、`duration_minutes`、
  `missing_minutes`、`confidence`、`truncated`、`truncated_edges`；
- `running_minutes` / `stopped_minutes` / `uncertain_minutes` 汇总。

## 本地开发与自测（不使用 Docker）

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

全天 1440 分钟、最短时长均为 30 的最坏规模（3600 个增强状态）在普通
CPU 上约 1.6 秒完成推断，进程峰值 RSS 约 65 MB（详见上节实测与
`benchmarks/run_benchmark.py`）。

## 已实现 / 已验证 / 未完成

**已实现**

- 分块检查点前向-后向：峰值工作内存由固定块大小（默认 128）与状态数
  决定，不再按窗口长度保存完整 alpha/beta/emission 矩阵；
- 精确重算，无近似、无降采样；缺测、首尾截断、逐分钟后验、区间标注与
  `log_likelihood` 语义不变，HTTP 请求/响应格式与字段不变；
- 只读模型缓存（带锁、容量受限）+ 请求内局部工作数组，保证并发隔离；
- 顺带修复旧实现的一个潜在缺陷：非对称最短时长下前驱度数可超过 8，
  旧的硬编码断言会导致 `/infer` 直接 500，现按实际度数定长。

**已验证**

- 31 项 unittest 全部通过（原 22 项 + 新增独立参考逐值对照、块大小
  不变性、并发隔离、前驱度回归共 9 项）；
- 与独立参考实现在多组小规模输入上逐值一致（容差 1e-10）；
- 与重构前稠密实现在全天最坏规模下**位级一致**（后验与
  `log_likelihood` 差异严格为 0，覆盖 `B=1..2000` 多种块大小）；
- 真实 uvicorn 服务端到端对照：示例请求、全天请求、全缺测、`T=1` 的
  JSON 响应与旧服务逐字节相同；16 个并发混合请求与串行结果零分歧；
- 全天实测内存（约 333 MB → 约 65 MB，降 81%）与耗时（约 1.3 s →
  约 1.6 s）。

**未完成 / 限制**

- 当前沙箱无 Docker 守护进程，`docker compose up --build` 未在本机
  实际构建运行；Dockerfile / compose 文件沿用原配置并已通过语法检查，
  需在有 Docker 的环境执行一次确认；
- 分块带来约 1 倍 alpha 重算，全天耗时增加约 20–30%（仍低于 2 秒，
  满足原有 10 秒性能断言）；若未来窗口上限继续增大，可考虑块大小
  自适应或进一步的检查点层级。
