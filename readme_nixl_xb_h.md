# 测试kv cache block

你这段代码是一个 **单元测试**，主要验证在 **prompt token 数量小于一个 KV cache block 大小时**，调度逻辑是否正确。下面逐行解析：

---

### 1. 测试目的说明

```python
"""
Test that we can handle case where prompt is < block.

In this case, the P worker will send empty remote_block_ids.
The D worker should not schedule an async read in this case,
since there is nothing to pull.
"""
```

* **场景**：输入 prompt 的 token 数量不足一个 block。
* **期望行为**：

  * P worker（Prefill worker）不会产生 remote\_block\_ids（因为根本没有完整的 block 可以 offload）。
  * D worker（Decode worker）也不应该发起异步读请求（async read），因为没有数据需要从远端拉取。

---

### 2. 初始化配置

```python
vllm_config = create_vllm_config()
scheduler = create_scheduler(vllm_config)
```

* 获取 vLLM 的配置对象 `vllm_config`。
* 用该配置初始化一个 `scheduler`（调度器）。

---

### 3. 定义测试场景：小于一个 block 的 token

```python
BLOCK_SIZE = vllm_config.cache_config.block_size
NUM_TOKENS = int(BLOCK_SIZE * 0.5)
```

* `BLOCK_SIZE`：KV Cache 中一个 block 能容纳多少 token。
* `NUM_TOKENS`：取一半 block 大小（即小于一个完整 block）。

---

### 4. 构造请求

```python
request = create_request(request_id=1,
                         num_tokens=NUM_TOKENS,
                         do_remote_prefill=True,
                         num_remote_blocks=0)
```

* `request_id=1`：请求编号。
* `num_tokens=NUM_TOKENS`：请求的 prompt token 数量不足一个 block。
* `do_remote_prefill=True`：说明该请求支持远端 prefill。
* `num_remote_blocks=0`：因为不足一块，所以 remote block 数量为 0。

---

### 5. 提交请求并执行调度

```python
scheduler.add_request(request)
scheduler_output = scheduler.schedule()
```

* 把请求交给调度器。
* 调用调度函数，获得调度结果 `scheduler_output`。

---

### 6. 验证调度结果

```python
kv_connector_metadata = scheduler_output.kv_connector_metadata
assert kv_connector_metadata is not None
assert isinstance(kv_connector_metadata, NixlConnectorMetadata)
assert len(kv_connector_metadata.requests) == 0
```

* 确认 `kv_connector_metadata` 存在，类型为 `NixlConnectorMetadata`。
* **关键点**：其中的 `requests` 列表长度为 0
  → 说明 D worker **没有发起任何 async read**（因为没有 remote block）。

---

### 7. 验证正常调度

```python
assert len(scheduler_output.scheduled_new_reqs) == 1
```

* 虽然没有远端 block，但这个请求本身仍然要正常进入调度。
* 所以 `scheduled_new_reqs` 中应包含该请求。

---

✅ **总结**：
这个测试用例确保了 **小于一个 block 的 prompt** 不会触发无意义的远端 KV 异步读取，但请求仍能被调度执行。
换句话说，这是在验证 **prefill 时 KV block 边界处理** 的正确性。

---

要不要我帮你画个 **调度流程图**（请求进入 → 判断 block 数量 → remote read vs regular schedule）？这样你可以更直观理解这个测试验证的逻辑。





# 注册nixl
```python
KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
    "NixlConnector")

```


# vllm调用nixl的软件栈
你问的 **“vLLM 调用 Nixl 的软件栈”**，其实就是在大模型推理/训练时，vLLM 如何把 **KV Cache 管理** 和 **远端内存访问** 跟 Nixl 结合起来。下面我给你梳理一下整个栈，从 **应用层 → 调度层 → Nixl 层 → 硬件层**：

---

## 1. 应用层（vLLM）

* **vLLM Core**：负责分布式推理的执行（prefill、decode、sampling 等），核心就是 **KV Cache 的存取**。
* **Block Manager / Scheduler**：vLLM 内部有一个 **block-based KV cache 管理器**，决定哪些 KV block 留在本地 GPU，高频访问，哪些 block 可以 offload 到远端。
* 当 prompt tokens < 1 block（就像你给的 test case），调度器不会产生 remote block 请求；而当 KV cache 超出本地显存容量时，就会触发 **remote block I/O**。

---

## 2. 中间件层（vLLM ↔ Nixl）

* vLLM 把 **远端 block 读/写** 抽象成 **KV Connector**，Nixl 只是其中的一种 backend。
* 这里的 `NixlConnectorMetadata` 就是调度器输出的元信息，告诉 Nixl：

  * 哪些 block 需要异步拉取
  * 哪些 block 可以丢弃
  * 哪些 block 要写回远端
* 这个接口层屏蔽了底层通信细节，vLLM 只关心 “我要从远端取哪些 block”。

---

## 3. Nixl 层（Memory Disaggregation Runtime）

* **Nixl** 提供了 **远端内存池 + 高速互连** 的抽象，通常基于 RDMA。
* 功能：

  * 提供远端内存分配（remote malloc/free）
  * 支持 block 粒度的 async read/write
  * 管理 **本地 GPU ↔ DPU ↔ 远端存储** 的数据流
* 在 vLLM 中，Nixl 就像一个 **远端 KV Cache Server**，专门存放溢出的 KV block。
* 一般实现上是基于 **libibverbs / RDMA verbs** 或者 **NVIDIA GPUDirect RDMA**，Nixl 负责把这些低层 API 封装成高层的 KV block I/O。

---

## 4. 硬件层

* **GPU**：计算 + 局部 KV Cache。
* **NIC/DPU (如 ConnectX-6 Dx, BlueField-3)**：负责 RDMA 传输，直接把远端 block 拉到 GPU 内存。
* **远端内存节点**：可能是 CPU 内存、大内存 GPU，或者专用的 **内存池节点**。

---

## 软件栈全景

```
用户 (LLM 推理请求)
        │
   vLLM Engine
        │
   ┌─────────────┐
   │ BlockManager│───调度KV cache，决定本地 vs 远端
   └─────────────┘
        │
   vLLM KV Connector
        │
   Nixl Runtime
        │
   RDMA / GPUDirect / Verbs
        │
   NIC / DPU (CX6, BF3)
        │
   远端内存池 (CPU DRAM / GPU HBM / CXL Mem)
```

---

## 核心要点

1. **vLLM 不直接调用 RDMA**，而是通过 `NixlConnector` 来访问远端 KV block。
2. **Nixl 负责 RDMA 管理**，屏蔽底层 QP、MR、CQ 等细节，暴露出 block 粒度的读写接口。
3. **DPU 承载 RDMA 请求**，保证 GPU 在需要时可以直接从远端拉数据，不需要 CPU 介入。
4. **调度逻辑在 vLLM**，**数据通路在 Nixl**。

---

要不要我给你画一个 **请求从 vLLM 到 Nixl 的数据流时序图**（比如 prefill → schedule → Nixl async read → DPU → GPU）？这样比软件栈分层更直观。




# 交互时序图
好，那我给你画一个 **时序图**，展示 vLLM 在 **需要远端 KV block 时**，调用 Nixl 的完整数据流：

---

## 📜 vLLM ↔ Nixl ↔ 硬件 的时序流程

```
用户请求 (Prompt / Decode)
        │
        ▼
   vLLM Scheduler
        │
        │ 判断：本地显存足够？
        │
        ├─ 是 → 直接调度 (走GPU本地KV Cache)
        │
        └─ 否 → 需要远端block
              │
              ▼
       NixlConnector
              │
              │ 发起 async read/write 请求 (block_id)
              ▼
         Nixl Runtime
              │
              │ 转换为 RDMA verbs (ibv_post_send/recv)
              ▼
        NIC / DPU (CX6-Dx / BF3)
              │
              │ 通过 RDMA 直接传输 (GPU RDMA / GPUDirect)
              ▼
       远端内存池 (CPU DRAM / GPU HBM / CXL Mem)
```

---

## 🔄 更详细的 **交互时序 (Async Read 示例)**

```
用户          vLLM-Scheduler   NixlConnector   Nixl Runtime    NIC/DPU    远端内存
 │                  │               │              │             │           │
 │ 请求推理         │               │              │             │           │
 │ ───────────────▶ │               │              │             │           │
 │                  │ 调度请求      │              │             │           │
 │                  │───生成block_id列表──────────▶│             │           │
 │                  │               │ 发起async read             │           │
 │                  │               │────────────▶ │             │           │
 │                  │               │              │ 转为ibv_post_send       │
 │                  │               │              │────────────▶│           │
 │                  │               │              │             │ RDMA读远端block
 │                  │               │              │             │──────────▶│
 │                  │               │              │             │           │
 │                  │               │              │             │ block数据返回
 │                  │               │              │             │◀──────────│
 │                  │               │              │ CQ通知完成  │           │
 │                  │               │◀──────────── │             │           │
 │                  │ KV block到GPU │              │             │           │
 │                  │               │              │             │           │
 │ 继续推理         │               │              │             │           │
 │ ◀─────────────── │               │              │             │           │
```

---

## 🚀 关键点

* **Scheduler**：决定是否触发远端读（block 粒度）。
* **NixlConnector**：抽象接口，把 “block\_id” 请求下发到 Nixl runtime。
* **Nixl Runtime**：封装 RDMA verbs（`ibv_post_send`、`ibv_post_recv`、`ibv_poll_cq`）。
* **NIC/DPU**：执行 RDMA 操作，直接搬运数据。
* **远端内存池**：可能是 CPU 大内存、另一台 GPU HBM，或 CXL 内存。

---

要不要我帮你把这个 **交互时序图画成 ASCII 之外的“框图+箭头图”（像论文里的数据流图）**？这样更直观，能看出 GPU ↔ DPU ↔ 远端内存的路径。


# vllm中的nixl
你问到 **nixl 在 vLLM 中的实现**，我来帮你梳理一下。

---

### 1. 背景

* **vLLM** 是一个高效推理引擎，核心优化点在于 **PagedAttention**、调度器、以及高效的 KV Cache 管理。
* 为了进一步提升分布式训练/推理中的通信效率，vLLM 引入了 **nixl**（NVIDIA Infiniband eXtensions Layer 的缩写，可以理解为一个 *RDMA 通信抽象层*）。
* nixl 的目的：在 vLLM 中屏蔽底层 **verbs API**（`ibv_post_send`、`ibv_poll_cq` 等）的复杂性，提供一个高性能、统一接口的 **通信层**，同时兼容 GPU Direct RDMA。

---

### 2. nixl 的设计思路

可以看成是 **vLLM 的网络栈实现**，和 NCCL 在 collective 通信上的角色有点类似，但更轻量、更面向 **point-to-point 异步传输**。

* **抽象层**
  nixl 将底层的 QP、CQ、MR 等抽象成“连接”和“内存区域”，提供面向消息/片段的接口。

* **零拷贝**
  nixl 在 GPU 上注册内存，直接走 GPUDirect RDMA，不需要 CPU 参与数据搬运。

* **异步调度**
  nixl 的 `send`/`recv` 接口是异步的，配合 vLLM 内部的 worker pool / scheduler，避免 block。
  它会先写入 `WQE`（work queue entry），再由 `completion queue` 驱动回调。

* **高并发 QP 管理**
  vLLM 的场景下，一个节点可能要和多个 peer 建立连接，nixl 提供了 QP 复用 / 连接池，减少 QP 爆炸。

---

### 3. 在 vLLM 中的作用

* **参数/激活分片传输**
  当 vLLM 做分布式推理时，需要在不同 rank 之间传输 KV Cache、输入 token embedding 等。
  nixl 负责这些跨节点数据的传输。

* **结合调度器**
  vLLM 的调度器 (scheduler) 在决定执行哪个 request 的时候，会调用 nixl 的接口来收/发远端 block，避免阻塞本地计算。

* **WorkerPool + nixl**
  你之前贴的 `WorkerPool` 代码，就是在管理异步任务队列。nixl 提供 `async read/write`，WorkerPool 把它和 request lifecycle 绑定。

---

### 4. 核心实现点

从 vLLM 代码来看，nixl 主要实现了几个模块：

1. **初始化 (context)**

   * 打开 RDMA device (`ibv_open_device`)
   * 创建 PD、CQ、QP
   * 注册 host/GPU 内存 (`ibv_reg_mr`)

2. **连接管理**

   * 通过 TCP/IB bootstrap 交换 QP 信息 (LID, QPN, PSN)
   * 修改 QP 状态 (INIT → RTR → RTS)

3. **数据收发**

   * `nixl_post_send()` → 封装 `ibv_post_send`
   * `nixl_poll_cq()` → poll 完成队列，触发回调
   * 支持 `RDMA_READ/WRITE`，在跨节点 KV cache 拉取时用得多

4. **错误处理 & 异常恢复**

   * 检查 QP/CQ 状态
   * 遇到网络错误时快速重连/abort

---

### 5. 为什么不用 NCCL？

* **NCCL 适合 collective** (all-reduce, all-gather 等)，但是 point-to-point、异步 granularity 不够灵活。
* vLLM 的数据传输更像是“拉取远端 KV 块”，量级小而频繁，用 RDMA verbs 更高效。
* nixl 就是做了 NCCL 没有覆盖的 **轻量 RDMA 通信层**。

---

### 6. 总结

在 vLLM 中，**nixl 就是一个 RDMA 通信抽象层**，负责：

* 建立连接、注册显存
* 异步收发数据
* 与调度器、worker pool 配合，保证 request lifecycle 内的 KV cache/参数能在不同节点之间高效流动

一句话总结：
**nixl 是 vLLM 的“网络驱动层”，专门针对大模型推理场景定制，比 NCCL 更灵活，比直接 verbs 更易用。**

---

要不要我帮你画一个 **“vLLM 调度器 – WorkerPool – nixl – RDMA device” 的结构图**？这样能直观看清 nixl 的位置。


在 vLLM 里，**nixl** 可以理解为一个专门为大模型推理场景做的 **RDMA 通信抽象层**。它的角色大概是：

* **屏蔽底层 verbs**：封装 `ibv_post_send` / `ibv_poll_cq` 等接口，统一成面向消息/内存块的 API。
* **高效传输**：支持显存注册，走 GPUDirect RDMA，零拷贝。
* **异步模型**：send/recv 都是异步的，和 vLLM 的调度器、worker pool 对接，避免阻塞。
* **连接管理**：建立 QP、CQ，维护连接池，减少 QP 爆炸问题。
* **主要用途**：跨节点传输 KV cache / 激活 / embedding 等，常用 `RDMA_READ/WRITE` 拉取远端数据。

相比 NCCL，nixl 更轻量、灵活：

* NCCL 偏向 collective（all-reduce, all-gather）；
* vLLM 的典型需求是“按需拉取远端 KV block”，量小而频繁，更适合 RDMA verbs。

一句话总结：
👉 **nixl 就是 vLLM 的网络驱动层，负责节点间高效的数据传输，比 NCCL 灵活，比直接写 verbs 简洁。**

要不要我给你画个 **调度器 – WorkerPool – nixl – RDMA device** 的框图，让 nixl 在 vLLM 架构里的位置更直观？




# 解读其连接器实现 nixl_connector.py
下面按你贴的这段代码，把 **vLLM 的 NixlConnector** 关键结构、调度/工作两端的职责、元数据流转、以及它如何决定“该不该去拉取远端 KV 块”这几件事讲清楚。重点会结合你之前提过的场景（prompt < block 时 P worker 发送 empty `remote_block_ids`、D worker 不应发起异步 RDMA 读取）。

---

# 总览：Nixl 在 vLLM 里的落点

* 这一组类把 **nixl RDMA 通道** 插进了 vLLM 的 **KV 迁移接口（KVConnectorBase\_V1）**。
* 代码分两侧：

  * **Scheduler 侧**（`NixlConnectorScheduler`）：决定“哪些请求需要从远端拉取/保存 KV”，把这些决策打包成 **ConnectorMetadata** 交给 Worker。
  * **Worker 侧**（`NixlConnectorWorker`，实现不在你这段里）：真正去注册 KV 显存、建立 RDMA 连接、发起 `RDMA_READ/WRITE`、以及可选的 Host 缓冲拷贝。
* **ZeroMQ 侧信道**（`side_channel_host/port`）仅用于交换 **Agent/内存地址等元信息**；真正的数据传输走 RDMA（nixl）。

---

# 关键对象与字段

### 1) 懒加载 nixl

```python
try:
    from nixl._api import nixl_agent as NixlWrapper
except ImportError:
    NixlWrapper = None
```

* 避免在没启用 nixl 时强行加载 RDMA 绑定。

### 2) 支持的 xPU 与缓冲设备

```python
_NIXL_SUPPORTED_XPUS = {
    "cuda": ("cuda", ),
    "tpu": ("cpu", ),
}
```

* CUDA 下支持直接用 **CUDA 缓冲**（GPUDirect RDMA）；TPU 则落回 **CPU 缓冲**。

### 3) `NixlAgentMetadata`

```python
class NixlAgentMetadata(msgspec.Struct):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    attn_backend_name: str
    kv_cache_layout: str
```

* 通过 **ZMQ 侧信道** 传输给对端（或 worker）的 **“握手+布局”元数据**：

  * `agent_metadata`：来自 `NixlWrapper`，用于 RDMA 层建立连接/交换 QP 之类的信息。
  * `kv_caches_base_addr / num_blocks / block_len / kv_cache_layout`：告诉对端 **KV cache 在内存/显存的布局与基址**，便于 RDMA 直接定位块。

### 4) 每个请求的元数据 `ReqMeta`

```python
@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int
```

* 一次请求在 **本地/远端** 各自涉及哪些 KV 块，以及对端的 **engine\_id / 地址**。
* `remote_block_ids` 可能为空（典型：**prompt < block**，没有需要从远端拉的块）。

### 5) `NixlConnectorMetadata`

* 这是 **Scheduler→Worker 的桥**，把要做的事汇总成三个哈希表：

  * `reqs_to_recv`：需要从远端 **拉取** KV 的请求
  * `reqs_to_save`：需要把本地 KV **保存**（写回对端/主机）的请求
  * `reqs_to_send`：需要 **发送** 的请求（带过期时间）
* 接口 `add_new_req(...)` 里 `load_remote_cache` 与 `save_to_host` 互斥，决定放进 `recv` 还是 `save`。

---

# NixlConnector 本体与 KV 布局

### 1) 角色拆分

```python
if role == KVConnectorRole.SCHEDULER:
    self.connector_scheduler = NixlConnectorScheduler(...)
elif role == KVConnectorRole.WORKER:
    self.connector_worker = NixlConnectorWorker(...)
```

* 一个进程以 **调度器** 或 **工作进程** 身份创建对应端的 connector。

### 2) KV Cache 布局要求

```python
@classmethod
def get_required_kvcache_layout(cls, vllm_config):
    if vllm_config.model_config.use_mla:
        return None
    logger.info_once("... KV cache layout to HND for better xfer performance.")
    return "HND"
```

* 若 **未使用 MLA**，强制将 KV 布局设为 **`HND`**（有利于 RDMA **整块直达、减少 stride/散布**）。
* 含义：Worker 注册的 KV 缓冲以这种布局暴露给对端，`num_blocks/block_len/base_addr` 都与之匹配。

---

# Scheduler 侧的核心逻辑

### 初始化

```python
self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
self.side_channel_port = (
    envs.VLLM_NIXL_SIDE_CHANNEL_PORT
    + dp_rank * tp_size
)
self.use_host_buffer = (kv_buffer_device == "cpu")
```

* **ZMQ 侧信道端口** = 基础端口 + `data_parallel_rank * tensor_parallel_size`，避免多并行度下冲突。
* 如果 KV 缓冲设备是 `cpu`，Worker 侧会走 **host 缓冲** 管线。

### 1) 决定能“额外匹配”的远端 token：`get_num_new_matched_tokens(...)`

核心用途：**远端预填（remote prefill）** 场景，统计“还能从远端 KV 直接加载多少 token（无需本地再算）”，并告知是否 **异步加载**。

伪代码（你贴的开头片段就这么干的）：

```python
if params and params.get("do_remote_prefill"):
    count = len(request.prompt_token_ids) - num_computed_tokens
    if count > 0:
        return (count, True)   # 还能加载这么多，并且会异步拉取
```

> ✅ **与你提到的 “prompt < block” 场景对齐**：
> 如果 `remote_block_ids` 为空（意味着远端并没有可拉的块），那么 `count` 会回落到 0 或者后续 `update_state_after_alloc` 不会把这个请求塞进 `reqs_to_recv`，**D/Worker 端也就不会发起 RDMA 读取**。

### 2) `update_state_after_alloc(request, blocks, num_external_tokens)`

* 在 **分配完本地 KV 块** 之后，根据请求的 `kv_transfer_params` 与 `blocks` 的对应关系，**把需要拉取/保存的请求编号 + 块 id** 填进：

  * `self._reqs_need_recv[req_id] = (request, local_block_ids)`
  * `self._reqs_need_save[req_id] = (request, local_block_ids)`
  * 以及 `self._reqs_need_send[req_id] = expire_time`

（实现细节在本文件后续或同目录其他文件里，这里你贴的片段还没到，但接口语义是明确的。）

### 3) `build_connector_meta(scheduler_output) -> NixlConnectorMetadata`

* 把 `_reqs_need_recv/_reqs_need_save/_reqs_need_send` 汇总打包成 **`NixlConnectorMetadata`**，交给 Worker（通过 vLLM 的调度/执行路径传下去）。

### 4) `request_finished(request, block_ids)`

* 请求结束时，决定是否需要把 **本地 KV 回写/保留**（例如供后续 remote prefill 复用），并返回给上层“是否需要保存”以及可能的额外参数。

---

# Worker 侧（从接口能看出的职责）

> Worker 具体实现不在你贴的段落里，但从这些方法能看清它干什么：

### 1) 注册 KV 显存

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    # 典型做法：用 nixl/GDR 注册显存为 RDMA MR
```

### 2) 可选的 host 缓冲拷贝能力

```python
def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp)
```

* 当 `kv_buffer_device == "cpu"` 时，Worker 用你传入的 `copy_blocks` 回调在 **GPU ↔ CPU** 之间做块级拷贝（比如保存到主机内存）。

### 3) 启动远端拉取

```python
def start_load_kv(self, forward_context: ForwardContext, **kwargs):
    # 读取上一步 Scheduler 打包的 NixlConnectorMetadata
    # 对每个 reqs_to_recv[req_id]：
    #   - 通过 ZMQ 获取对端 NixlAgentMetadata（base_addr、layout）
    #   - 由 NixlWrapper 发起 RDMA_READ，把远端 block 直接 dma 到本地 KV 块
```

### 4) 查询完成态

```python
def get_finished(self, finished_req_ids: set[str]) -> (set[str], set[str])
```

* 返回 **完成接收** 与 **完成发送/保存** 的请求集合，供引擎上层推进调度。

### 5) 保存 KV（host 缓冲路径）

```python
def wait_for_save(self):
    if self.use_host_buffer and self.copy_blocks:
        self.save_kv_to_host(self._connector_metadata)
```

* 若配置为 **CPU 缓冲**，在这里触发 **显存→内存** 的分块拷贝与（可能的）RDMA 发送。

> 另外可以注意到：
>
> ```python
> def wait_for_layer_load(self, layer_name): pass
> def save_kv_layer(...): pass
> ```
>
> * Nixl 路径下不做 **按层** 的保存/加载，而是按 **块** 粒度一次性处理，这与它的 RDMA 直达设计相符。

---

# 关于 “prompt < block” 的具体行为

* 当 **prompt token 数 < block\_size**，通常 **不会产生任何 `remote_block_ids`**（远端没有完整块可复用）。
* 这会导致：

  1. `get_num_new_matched_tokens(...)` 返回 “可加载 token 数为 0 / 不需要异步加载”，或
  2. `update_state_after_alloc(...)` **不把该请求塞进** `_reqs_need_recv`，
  3. `build_connector_meta(...)` 里 **`reqs_to_recv` 不含该请求**，
  4. Worker 侧的 `start_load_kv(...)` 自然 **不会发起 RDMA\_READ**。
* 这正是你提到的期望行为：“**D worker 不应调度异步读**”。

---

# 小结（把主线串一下）

1. **Scheduler** 判定这个请求是否要“远端预填/拉块”，并记录需要拉取/保存的 **块 id 列表**。
2. 把这些决策打包成 **`NixlConnectorMetadata`** 给 **Worker**。
3. **Worker** 通过 **ZMQ** 交换 **NixlAgentMetadata**（对端 QP/基址/布局），再用 **nixl（RDMA）** 直接把远端块 **DMA 到本地 KV**。
4. 如果使用 **CPU 缓冲**，按需做 **GPU↔CPU** 的块拷贝与保存。
5. **`remote_block_ids` 为空**（比如 prompt < block）时，**整个 RDMA 拉取路径不会被触发**。

如果你愿意，我可以继续把 **`get_num_new_matched_tokens / update_state_after_alloc / build_connector_meta / request_finished`** 的完整调用链配上“输入输出样例”和“典型时序（prefill→decode→finish）”，这样你调试时能直接对照日志与状态。






