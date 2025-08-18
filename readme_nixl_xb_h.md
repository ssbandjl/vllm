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



