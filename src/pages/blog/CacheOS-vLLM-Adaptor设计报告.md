# CacheOS-vLLM Adaptor 设计报告（A5 非 MTP 最小闭环）

> 本文是设计文档，不包含代码实现。代码基线是
> `vLLM-ascend-DSA` 的 `origin/vllm-ascend-v0.23.0-custom@6af99b372`、
> A5 算子的 `origin/ops_li_update_a5@7d400ea8d`，以及当前工作区的
> `KVAF`。详细 A5 ABI 见 [LIDU-A5算子实现报告](./LIDU-A5算子实现报告.md)。
>
> 范围锁定为单 Token Decode，`K=2048`，不支持 MTP。本文的最小闭环
> 只代替当前 v0.23 链路中“KSC 回载 + 满块 DRAM dump”的部分，
> 不重新设计 Lightning Indexer 和 SFA 算法。

## 0. 先看当前要打通的完整流程

```text
图外/请求控制面
  vLLM request + layer + logical MLA block
    → Adaptor 生成两类稳定 Key（NOPE block / ROPE block）
    → Adaptor 构造 kvaf::AccessTablePrepareRequest
    → KVAF Runtime::PrepareAccessTable / GetAccessTableView
    → Runtime 编排 MetaManager，持有 Key→DDR 地址的稳定映射
    → Adaptor 持有固定容量的 device table / mask / indices scratch

当前 layer 的 Decode 数据面（同一 NPU stream）
  vLLM 产生 query / Indexer K / weights / metadata
                    │
                    ▼
  VllmA5Bridge（A5 前置适配，不是 KVAF Adaptor 内核）
                    │
                    ▼
  LIDU A5
    → topk_index [B,1,2048]  original position，miss 前缀
    → topk_slots [B,1,2048]  目标 resident logical slot
    → miss_count [B]
    → cache_slots 已原地更新
                    │
                    ▼
  CacheOS-vLLM Adaptor
    → mask[b,k] = (k < miss_count[b])
    → topk_index p 翻译为 kvaf::BatchGetRequest.indices
    → topk_slots s 经 resident block_table 翻译为 HBM physical token row
    → 构造 NOPE/ROPE 两个 kvaf::BatchGetRequest
    → KVAF Runtime::BatchGet(NOPE)：原址写 resident NOPE HBM
    → KVAF Runtime::BatchGet(ROPE)：原址写 resident ROPE HBM
    → Runtime 内部编排 AccessOperator 完成 DDR→HBM
    → Adaptor 消费 Runtime Status 和 caller stream 顺序语义，不消费算子私有输出
                    │
                    ▼
  SFA handoff
    → sparse_indices = topk_slots [B,1,2048]
    → tail_info      = Adaptor/vLLM 管理的 dense-tail 信息
    → resident NOPE/ROPE HBM 是 Runtime::BatchGet 的原地结果
    → SFA-Offload 从已就绪的 resident HBM 读 NOPE/ROPE
                    │
                    ▼
  当 vLLM 写满一个 128-token MLA block
    → Adaptor 构造 NOPE/ROPE 两个 kvaf::BatchPutRequest
    → KVAF Runtime::BatchPut(NOPE full block)
    → KVAF Runtime::BatchPut(ROPE full block)
    → Runtime 内部执行 PreparePut→AccessOperator→Commit/Abort
    → Adaptor 根据两个 Runtime Status 标记 remote-readable
    → 下一 decode step 才允许 A5 miss 引用该块
```

本阶段的核心结论是：**Adaptor 的唯一系统对接面是 KVAF Runtime。**
它不直接对接 MetaManager、AccessOperator、AIV kernel 或远端地址。具体为：

1. **Adaptor 消费 vLLM/A5 输出，生成 KVAF Runtime 公共请求。**
2. **Adaptor 消费 Runtime 返回的 Status、caller stream 顺序语义和原地 HBM 结果，生成 SFA 输入。**
3. **A5 决定 resident slot，Adaptor 不再实现第二套淘汰算法。**
4. **Adaptor 为 `kvaf::BatchGetRequest` 生成的 mask 是 A5 miss 前缀契约的物化。**
5. **DDR 写入粒度为 128-token 满块，miss 回载粒度为 1 token。**
6. **NOPE/ROPE 保留为两个 plane，MVP 调用两次 Runtime::BatchGet，原地写回
   vLLM resident paged HBM，不创建额外 gather tensor。**
7. **SFA 只消费 Adaptor handoff，不感知 Runtime request、DDR 地址或 KVAF Key。**

Adaptor 的唯一职责定义为：

```text
Adaptor(vLLM_outputs, vLLM_context)
  = EncodeRuntimeRequests(vLLM_outputs, vLLM_context)
  + InvokeKvafRuntime(runtime_requests)
  + DecodeRuntimeResults(runtime_status, in_place_outputs, dependency)
  + BuildSfaInputs(resident_cache, topk_slots, tail_info, block_table)
```

| 边界 | 输入 | Adaptor 处理 | 输出 |
|---|---|---|---|
| vLLM → Adaptor | A5 选择输出 + request/layer/cache metadata | Key、index、mask、output row 转换 | KVAF Runtime 公共 request |
| Adaptor → KVAF | `kvaf::AccessTablePrepareRequest/BatchGetRequest/BatchPutRequest/...` | 仅通过 `KvafRuntimeClient` 调用 Runtime | Runtime Status/handle/view + caller-owned output 原地变化；完成顺序由 caller stream 保证 |
| KVAF → Adaptor | Runtime 公共结果 | 状态收敛、就绪性建立、SFA 视图组装 | SFA 所需 resident cache/block table + sparse indices/tail |

这个边界是后续实现和 code review 的首要验收条件。

## 1. 设计目标和非目标

### 1.1 MVP 目标

- 接收 A5 的 `topk_index/topk_slots/miss_count`；
- 把 vLLM/A5 输出翻译为 Runtime 公共类型
  `kvaf::AccessTablePrepareRequest/kvaf::BatchGetRequest/kvaf::BatchPutRequest`；
- 为每行 2048 个固定位置生成 Runtime 所需的 graph-friendly
  `indices/mask/output_indices`；
- 通过 `Runtime::BatchGet` 使 hit 不传输、miss 从 KVAF 管理的 DDR
  原地回填 resident HBM；
- 同时回填 MLA NOPE `[512]` 和 ROPE `[64]` 两个 plane；
- 在满 128-token block 后通过 `Runtime::BatchPut` 将两个 plane 写入 DDR，
  Runtime 内部在 KVAF MetaManager 中建立稳定 Key 寻址映射；
- 将 Runtime Status、caller stream 顺序和原地 HBM 结果收敛为 SFA handoff，
  给 SFA 提供 `topk_slots + tail_info + resident cache/block table`；
- 给后续图模式、异步 BatchPut、KVIP、MTP 预留边界，但 MVP 不实现它们。

### 1.2 非目标

- 不支持 MTP 或一个 request row 多个 query token；
- 不修改 A5 Top-K 算法、victim 选择或 score 语义；
- 不让 KVAF Runtime 理解 request、layer、prefix、SFA 或 vLLM block table；
- 不让 Adaptor 保存或计算远端 UBA 裸地址；
- 不让 Adaptor 直接调用 `MetaManager`、`AccessOperator::ExecuteBatchGet`、
  `AccessOperator::ExecuteBatchPut` 或 AIV kernel；
- 不做跨请求 Prefix KV 共享；MVP Key 是 request-generation scoped；
- 不设计分布式 Meta 协议、多卡一致性或 KVIP 路由；
- 不在当前阶段融合 NOPE/ROPE BatchGet 或实现 Megakernel。

## 2. 必须遵守的现有代码事实

### 2.1 KVAF 责任边界

`KVAF/docs/kvaf_architecture_design.md` 已经将语义边界定义清楚：

| 内容 | 所有者 |
|---|---|
| request/layer/block/plane 语义和 Key 生成 | CacheOS-vLLM Adaptor |
| A5 输出到 indices/mask/output_indices 的翻译 | CacheOS-vLLM Adaptor |
| Key → DDR 地址映射和地址生命期 | KVAF MetaManager |
| `PrepareAccessTable/GetAccessTableView/BatchGet/BatchPut/Release/Free` 公共入口 | KVAF Runtime |
| 真实 UDMA 读写 | KV Access Operator |
| resident HBM tensor 和 HBM block table | vLLM |
| Top-K/victim/resident logical slot 更新 | A5 |
| sparse slots + tail 的 attention | SFA-Offload |

Adaptor 可以保存 AccessTable handle 和相对索引，但不得绕开 Runtime
缓存 Key 对应的裸地址。

### 2.2 Adaptor 对接的 Runtime 公共接口

当前 C++ 公共边界以 `KVAF/runtime/include/kvaf/runtime/runtime.h` 为准：

```cpp
kvaf::Runtime::Initialize(const kvaf::RuntimeInitRequest&)
kvaf::Runtime::PrepareAccessTable(const kvaf::AccessTablePrepareRequest&,
                                  kvaf::AccessTableHandle*)
kvaf::Runtime::GetAccessTableView(const kvaf::AccessTableHandle&,
                                  kvaf::AccessTableView*)
kvaf::Runtime::ReleaseAccessTable(const kvaf::AccessTableHandle&)
kvaf::Runtime::BatchGet(const kvaf::BatchGetRequest&,
                        const kvaf::BatchGetExecutionOptions&)
kvaf::Runtime::BatchPut(const kvaf::BatchPutRequest&,
                        kvaf::BatchPutExecutionOptions)
kvaf::Runtime::BatchFree(const kvaf::BatchFreeRequest&)
kvaf::Runtime::Shutdown()
```

Adaptor 只依赖上述 Runtime 对象和 `KVAF/include/kvaf/requests.h` 中的公共
request/view/handle 类型。`Runtime::BatchGet` / `Runtime::BatchPut` 是 **Runtime 公共
方法名**；这不等于 Adaptor 直接调用 BatchGet/BatchPut 数据面算子。

严格的边界是：

```text
Adaptor
  → Runtime::BatchGet(kvaf::BatchGetRequest)
      → Runtime 校验 AccessTableHandle 和 buffer
      → Runtime 组织 BatchGetExecRequest
      → AccessOperator::ExecuteBatchGet（Runtime 内部）

Adaptor
  → Runtime::BatchPut(kvaf::BatchPutRequest)
      → MetaManager::PreparePut（Runtime 内部）
      → AccessOperator::ExecuteBatchPut（Runtime 内部）
      → MetaManager::CommitPut / AbortPut（Runtime 内部）
```

### 2.3 Runtime 输入/输出与 Adaptor 处理关系

| Runtime 公共方法 | Adaptor 生成的 Runtime 输入 | Runtime 对 Adaptor 的输出 | Adaptor 后处理 |
|---|---|---|---|
| `PrepareAccessTable` | `flat_keys` | `Status + AccessTableHandle` | 保存 handle，用于后续 Runtime 请求 |
| `GetAccessTableView` / NPU materialize facade | `AccessTableHandle` | `Status + AccessTableView/DeviceAccessTable` | 保存固定 device view，不解析裸地址 |
| `BatchGet` | `kvaf::BatchGetRequest + kvaf::BatchGetExecutionOptions` | `Status`；数据结果原地写入 `request.output`；默认完成顺序由 caller stream 保证 | 检查提交 Status 和 stream 顺序，将已就绪 resident cache 放入 SFA handoff |
| `BatchPut` | `kvaf::BatchPutRequest + kvaf::BatchPutExecutionOptions` | `Status`；成功表示数据搬运和 Meta commit 完成 | 只在 NOPE/ROPE 都成功后标记 remote-readable |
| `ReleaseAccessTable` | `AccessTableHandle` | `Status` | 结束 epoch，不删除 KV |
| `BatchFree` | `BatchFreeRequest{keys}` | `Status` | 更新 request 生命周期，不自行回收 DDR 地址 |

这里最容易混淆的是 `Runtime::BatchGet` 输出：它不返回一个新 KV
Tensor。返回值是 `Status`，真实 KV 字节是 Runtime 按 `BatchGetRequest.output`
原地写入 vLLM resident HBM。Adaptor 对 Runtime 结果的“处理”是：

1. 确认 Runtime 提交 Status 成功，并以同 stream 顺序或显式同步满足 SFA 前置条件；
2. 保留 Runtime 已原地写入的 resident cache tensor；
3. 将 `topk_slots` 转成 `sparse_indices`，合并 `tail_info`、resident cache 和
   resident block table，形成 SFA handoff。

### 2.4 Runtime 内部 BatchGet 数据面限制（非 Adaptor 对接接口）

`KVAF/access_operator/demo/aiv_batch_get/include/batch_get_abi.h` 和 kernel 实现的
有效项 `i` 语义是：

```text
src = address_table[indices[i]]
dst = output + output_indices[i] * block_bytes

UDMA_READ(src, dst, block_bytes)
```

| 张量/标量 | 当前 ABI |
|---|---|
| `address_table` | `uint64[table_capacity]` |
| `valid_table` | `uint8[table_capacity]` |
| `indices` | `int32[request_capacity]` |
| `mask` | `uint8[request_capacity]` |
| `output_indices` | `int32[request_capacity]` |
| `output` | caller-owned contiguous HBM |
| `block_bytes` | 一次 launch 中固定 |

当前 ABI **没有 source block 内的 token offset**。这个事实决定了：如果
KVAF 中一个 Key 是 128-token 满块，现有 Runtime 公共 `BatchGetRequest`
还不能表示“读该块的第 37 个 token”。该问题必须以 Runtime
能力扩展形式解决；Adaptor 不得绕开 Runtime 直接给 AIV kernel 填地址。

### 2.5 当前 Runtime BatchPut/AccessTable 的事实

- BatchPut 输入是 caller-owned 连续 buffer + Host `input_offsets/lengths`；
- BatchPut V1 是 synchronous eager，当前不支持入图；
- Runtime 在数据搬运成功后才 `CommitPut`，失败会 `AbortPut`；
- `PrepareAccessTable` 是 Host 控制面操作，不入图；
- 已生成的 AccessTable 不会在新 BatchPut 成功后自动刷新；
- 当前 NPU BatchGet 默认只 enqueue 到 caller stream，不隐式同步；
- 当前 NPU AccessTable 对每次 launch 要求所有 valid entry 具有相同的
  effective `block_bytes`。

## 3. 总体架构和模块分工

```text
vLLM SFA Frontend
  │
  ├─ VllmA5Bridge
  │    ├─ v0.23 metadata → A5 input
  │    ├─ stable request row / first-fill / dense-tail
  │    └─ caller-owned A5 outputs（目标 ABI）
  │
  ├─ LIDU A5
  │    └─ A5SelectionView
  │
  └─ CacheOSVllmAdaptor
       ├─ VllmOutputAdapter
       │    ├─ SelectionNormalizer / MissMaskBuilder
       │    ├─ SourceIndexTranslator
       │    └─ ResidentOutputTranslator
       ├─ AccessTableEpochManager
       ├─ KvafRuntimeClient          # 唯一 KVAF 对接面
       │    ├─ Prepare/Release AccessTable
       │    ├─ Runtime::BatchGet x 2
       │    ├─ Runtime::BatchPut x 2
       │    ├─ Runtime::BatchFree
       │    └─ KVAF Runtime
       │         ├─ MetaManager（Runtime 内部）
       │         └─ AccessOperator（Runtime 内部）
       ├─ RuntimeResultAdapter
       ├─ FullBlockLifecycleManager
       └─ SfaHandoffBuilder → SFA-Offload
```

### 3.1 `VllmA5Bridge`

这是 A5 与 v0.23 之间的前置适配层，不属于 KVAF Runtime。它负责：

- 把 v0.23 `q_li/weights/indexer_cache/indexer_block_table` 组成 A5 ABI；
- 对 `weights` 做 contiguous 规范化；
- 将 batch row 稳定映射到 request resident-state row；
- 完成 first-fill 和 resident budget 初始化；
- 仅向 A5 暴露已 remote-committed 的 sparse candidate prefix；
- 另行生成 `[tail_slot_start,tail_token_count]`；
- 目标上使用 caller-owned A5 out ABI，避免 graph 中 `at::empty`。

不能用 Adaptor 后置的 Runtime read 转换逻辑替代这一层，否则 A5 与 v0.23
的 row mode、tail 和 first-fill 差异会被错误隐藏。

### 3.2 `VllmOutputAdapter`

该模块只消费 vLLM/A5 暴露的张量和元数据，不调用 KVAF 内部模块。
其中 `SelectionNormalizer/MissMaskBuilder` 不修改 A5 结果，只把
`[B,1,K]` 规范为 Runtime `BatchGetRequest` 所需的连续 `[B,K]`/flat view，
并在固定容量 scratch 中生成 miss mask。

### 3.3 `SourceIndexTranslator`

将 A5 original position `p` 翻译为 Runtime-provided DeviceAccessTable 的相对 entry，
不接触远端裸地址。

### 3.4 `ResidentOutputTranslator`

将 A5 resident logical slot `s` 通过 vLLM resident MLA block table 翻译为
`kvaf::BatchGetRequest.output` 中的 physical token row。

### 3.5 `AccessTableEpochManager`

管理每个 layer/plane 的 Key 集合、AccessTableHandle、固定地址 DeviceAccessTable
和可见 epoch。所有 handle/view 都通过 `KvafRuntimeClient` 调用 Runtime 获得；
新块只有在 `Runtime::BatchPut` 成功后才进入新 epoch。

### 3.6 `KvafRuntimeClient`

这是 Adaptor 内部唯一允许依赖 KVAF 的模块。它只持有 `kvaf::Runtime&`，
并只调用 Runtime 公共方法。它负责：

- 将 Adaptor 生成的 Key 组装为 `kvaf::AccessTablePrepareRequest`；
- 调用 `Runtime::PrepareAccessTable/GetAccessTableView/ReleaseAccessTable`；
- 将相同 `indices/mask/output_indices` 组装为 NOPE/ROPE 两个
  `kvaf::BatchGetRequest`，分别调用 `Runtime::BatchGet`；
- 将满块 Key/buffer/offset/length 组装为 `kvaf::BatchPutRequest`，调用
  `Runtime::BatchPut`；
- 在 request 结束时调用 `Runtime::BatchFree`。

它不包含 `MetaManager*`、`AccessOperator*`、kernel object 或远端地址。

### 3.7 `RuntimeResultAdapter`

统一消费 Runtime 公共方法的结果：

- `PrepareAccessTable`：接收 handle 并更新 epoch state；
- `BatchGet`：接收提交 `Status`，并用 caller stream 顺序或显式同步识别
  `request.output` 已原地就绪；
- `BatchPut`：只在两个 plane 都成功时将 logical block 标记为
  remote-readable；
- 任何失败：转换为 Adaptor request/layer 状态，不读 Operator 私有 status。

### 3.8 `FullBlockLifecycleManager`

在 vLLM MLA block 满 128 token 后，按 layer 将 NOPE/ROPE 整块写入 KVAF，
它只调用 `KvafRuntimeClient`，并且只在两次 `Runtime::BatchPut`
都成功后将逻辑块标记为 remote-readable。

### 3.9 `SfaHandoffBuilder`

在 `RuntimeResultAdapter` 证明两个 plane 已提交/就绪后，建立 SFA-Offload 需要的
resident 视图。SFA 不看 `miss_count`、`topk_index`、Runtime request、KVAF Key
或 AccessTable。

## 4. 参数、shape 和设计级接口

### 4.1 固定符号

| 符号 | 含义 | MVP 值 |
|---|---|---:|
| `B` | 当前活跃 Decode request 行数 | 动态，`B<=Bmax` |
| `Bmax` | graph/scratch 最大行数 | 配置值 |
| `K` | A5 sparse Top-K | 2048 |
| `BS` | MLA block token 数 | 128 |
| `D_nope` | 每 token NOPE 元素数 | 512 |
| `D_rope` | 每 token ROPE 元素数 | 64 |
| `P` | A5 position 容量 | 262144 |
| `M` | 每请求最大 remote logical block 数 | `ceil(P/BS)=2048` 上界 |

### 4.2 A5 核心选择输入

```text
A5SelectionView
├─ topk_index : int32 [B,1,2048]  NPU contiguous/view
├─ topk_slots : int32 [B,1,2048]  NPU contiguous/view
└─ miss_count : int32 [B]         NPU contiguous
```

Adaptor 不再接收 `query/key/weights`。这些属于 A5 的输入，不是
KVAF Runtime 请求的输入。

### 4.3 Decode 调用的支撑上下文

A5 三个输出是选择核心，但不是完整寻址上下文。Adaptor 还需要：

```text
AdaptorDecodeContext
├─ selection: A5SelectionView
├─ layer_id                    Host scalar / stable layer context
├─ active_B                    Host scalar or graph-bucket constant
├─ request_pool_entries     int32 [B]
├─ row_modes                int32 [Bmax]
├─ remote_sparse_tokens     int32 [B]
├─ remote_block_ordinal     int32 [pool_rows,M]
├─ resident_block_table     int32 [B,resident_max_blocks]
├─ resident_nope_cache      BF16/FP16 [N_hbm,128,1,512]
├─ resident_rope_cache      BF16/FP16 [N_hbm,128,1,64]
├─ nope_access_table        Runtime-provided stable device AccessTable
├─ rope_access_table        Runtime-provided stable device AccessTable
├─ tail_info                int32 [B,2]
└─ stream                   current NPU stream
```

`remote_block_ordinal` 是 Adaptor 管理的逻辑表，值是 AccessTable 中的相对
block ordinal，不是地址。`-1` 表示该 logical block 尚未 remote-committed。

### 4.4 Adaptor 固定 scratch

```text
AdaptorDeviceScratch（所有张量地址在 graph 期间不变）
├─ indices        int32 [Bmax,2048]
├─ mask           uint8 [Bmax,2048]
├─ output_indices int32 [Bmax,2048]
└─ optional debug int32 [Bmax] / status
```

`KvafRuntimeClient` 在传给当前 `NpuRuntime.batch_get` 前取无拷贝的一维 view：

```text
indices_flat        int32 [Bmax*2048]
mask_flat           uint8 [Bmax*2048]
output_indices_flat int32 [Bmax*2048]
```

这是因为当前 NPU Runtime facade 明确要求这三个张量是 rank 1。C++
`Runtime::BatchGet` 也将它们视为连续 flat buffer，
`request_capacity=Bmax*2048`。非活跃 row 和 hit 项的 `mask=0`，从而保持
graph shape 固定。

### 4.5 Runtime 请求输入

```text
RuntimeReadInputs
├─ NOPE BatchGetRequest
│    ├─ access_table / device_access_table
│    ├─ indices / mask / output_indices
│    ├─ output = resident_nope_cache.view(N_hbm*128,512)
│    └─ output_block_bytes = 512 * element_size
├─ ROPE BatchGetRequest
│    ├─ access_table / device_access_table
│    ├─ indices / mask / output_indices
│    ├─ output = resident_rope_cache.view(N_hbm*128,64)
│    └─ output_block_bytes = 64 * element_size
└─ BatchGetExecutionOptions{stream,synchronize}
```

这两个请求都由 `KvafRuntimeClient` 传给 `Runtime::BatchGet`，不传给
AccessOperator。Runtime 负责进一步组织数据面执行请求。

### 4.6 Runtime 结果和 SFA handoff

```text
RuntimeReadResult（逻辑表达）
├─ nope_submit_status
├─ rope_submit_status
├─ ready_dependency       # 当前为 caller stream 顺序；未来可扩展为 Event/device gate
├─ resident_nope_cache  # Runtime 已原地写入
└─ resident_rope_cache  # Runtime 已原地写入

SfaHandoff
├─ sparse_indices       = topk_slots int32 [B,1,2048]
├─ tail_info                         int32 [B,2]
├─ resident_nope_cache               BF16/FP16 [N_hbm,128,1,512]
├─ resident_rope_cache               BF16/FP16 [N_hbm,128,1,64]
├─ resident_block_table              int32 [B,resident_max_blocks]
└─ ready_dependency                  same-stream order or Event
```

`tail_info[b] = [tail_slot_start,tail_token_count]`。A5 不输出这个张量，
它由 `VllmA5Bridge`/vLLM request layout 生成。

`SfaHandoff` 是逻辑契约，不要求实现一定新建一个 Python/C++ 对象。
当前 v0.23 可继续用 `DSAOffloadSelectionOutput(sparse_indices,tail_info)` 承载选择
部分，resident cache/block table 由现有 SFA 调用参数传入。关键是 Adaptor
必须在 Runtime 完成依赖建立后才允许该 SFA 调用执行。

当前 Runtime 没有返回独立的 completion handle。`synchronize=false` 时，
`Runtime::BatchGet` 返回成功只表示已提交到 caller stream；Adaptor 依靠同一 stream
上的 SFA 后续操作保证先读后用。correctness bring-up 可使用
`kvaf::BatchGetExecutionOptions{synchronize=true}`；跨 stream 或图内失败 gate 属于
后续 Runtime 公共能力扩展。

## 5. DDR 对象、Key 与地址模型

### 5.1 为什么按 plane 存两个满块

v0.23 的 resident MLA cache 是两个独立 tensor：

```text
NOPE block : [128,1,512]
ROPE block : [128,1, 64]
```

BF16/FP16 都是 2 bytes，因此每 layer：

```text
NOPE bytes = 128 * 512 * 2 = 131072 B = 128 KiB
ROPE bytes = 128 *  64 * 2 =  16384 B =  16 KiB
total      =                         144 KiB
```

当前 BatchGet 只有一个 output buffer 和一个固定 `block_bytes`。MVP 把两个
plane 分开存储和读取，可以不增加多输出 kernel ABI，也不需要中间
pack/unpack tensor。

### 5.2 MVP Key

设计上的 Key 必须唯一标识一个完整 plane block：

```text
KVBlockKey = {
  namespace_id,
  model_instance_id,
  request_generation_id,
  layer_id,
  logical_block_id,
  plane,                 # NOPE or ROPE
  layout_version,
  dtype,
}
```

MVP 使用 `request_generation_id`，避免 request id 复用导致旧 KV 被误命中。
未来如需 prefix sharing，可以把 request-scoped 部分换成 content/prefix identity，
KVAF Runtime 接口无需因此变更。

### 5.3 三种解决“满块存、单 token 取”的方案

| 方案 | 做法 | 优点 | 代价 | 决策 |
|---|---|---|---|---|
| A. Runtime token-projected AccessTable | 扩展 `AccessTablePrepareRequest`，由 Runtime 将每个 full-block lookup 结果展开为 128 个 token entry | Adaptor 仍只对接 Runtime；现有 `BatchGetRequest` 不变 | Device table 放大 128 倍；Runtime AccessTable state/materialize 需扩展 | **MVP 采用** |
| B. Runtime BatchGetRequest 增加 source offset | AccessTable 仍按 block，Adaptor 给 Runtime 每个 request 携带 token byte offset | 表小，适合生产 graph | 需扩展 Runtime 公共 request，并由 Runtime 向下适配 Operator | 后续优化首选 |
| C. KVAF 按 token 存 Key | 满块触发时写 128 个 token object | 完全复用当前 BatchGet | Key/Meta/DDR allocation 膨胀，破坏整块连续性 | 不采用 |

MVP 选 A 的原因是它可以先验证 A5→mask→BatchGet→SFA 这条数据链，
同时保留 KVAF Meta 与 BatchPut 的 full-block 语义。它是 correctness MVP，不是
最终的大规模 Device AccessTable 形态。

### 5.4 token-projected AccessTable 的最小 Runtime 公共接口扩展

不让 Adaptor 在获得 `AccessTableView` 后自行读取/派生地址。推荐将 projection
作为 Runtime `PrepareAccessTable` 公共请求的可选字段：

```text
AccessTablePrepareRequest
├─ flat_keys: vector<KeyId>           # 仍是 full-block keys
└─ projection: optional EntryProjection
     ├─ entries_per_key = 128
     ├─ entry_bytes     = D_plane * element_size
     └─ entry_stride    = D_plane * element_size
```

Runtime 仍使用 full-block keys 调用 MetaManager `LockKeys/BatchLookup`。在得到
block address/length 后，Runtime 在自己的 `AccessTableState` 中建立投影：

```text
projected_address[j*128+t] = block_base_address[j] + t*entry_stride
projected_length[j*128+t]  = entry_bytes
projected_valid[j*128+t]   = block_valid[j]
                             && block_length[j] >= (t+1)*entry_bytes
```

`Runtime::GetAccessTableView` 返回的是投影后的 Runtime-owned view；NPU Runtime facade
再将该 view materialize 为 DeviceAccessTable。MetaManager 仍只锁住原始 full-block
Key，Adaptor 始终看不到 `block_base_address`。

因此 MVP 的 Runtime 变更面是：

- `requests.h`：为 `AccessTablePrepareRequest` 增加可选 projection 契约；
- `Runtime::PrepareAccessTable`：检查投影范围，保存投影后
  address/length/valid；
- `Runtime::GetAccessTableView`：继续返回 Runtime-owned view，但 view 的 entry
  已是 token 粒度；
- NPU Runtime facade：将 Runtime view 写入 caller-owned 固定 DeviceAccessTable
  buffer，以满足 graph 地址稳定性；
- `Runtime::BatchGet`：继续校验 uniform valid length，此时统一长度就是
  token bytes；
- 现有 Runtime `BatchGetRequest` 公共字段保持不变。

是否需要修改 AIV kernel 是 Runtime/AccessOperator 的向下实现决策，不是 Adaptor
与 Runtime 之间的接口。方案 A 下现有 AIV kernel 可以继续消费投影后的
device address table。

### 5.5 Source 相对索引

Adaptor 对每个 miss `p` 计算：

```text
logical_block = p // 128
token_offset  = p % 128
pool_row      = request_pool_entries[b]
block_ordinal = remote_block_ordinal[pool_row, logical_block]

indices[b,k] = block_ordinal * 128 + token_offset
```

`block_ordinal` 是 NOPE/ROPE 两张 AccessTable 共用的逻辑顺序。两张表的 Key
序列都按同一 `(request_generation,layer,logical_block)` 顺序构建，只有 `plane`
不同，因此可以共用 `indices`。

## 6. resident HBM 目标寻址

A5 返回的 `s` 是 resident logical slot，不是 HBM physical row。Adaptor 必须通过
vLLM 当前 layer 的 resident block table 翻译：

```text
logical_resident_block = s // 128
resident_token_offset  = s % 128
physical_resident_block = resident_block_table[b,logical_resident_block]

output_indices[b,k]
  = physical_resident_block * 128 + resident_token_offset
```

对 NOPE BatchGet：

```text
output             = resident_nope_cache.view(N_hbm*128,512)
output_rows        = output.shape[0] = N_hbm * 128
output_block_bytes = 512 * element_size
```

对 ROPE BatchGet：

```text
output             = resident_rope_cache.view(N_hbm*128,64)
output_rows        = output.shape[0] = N_hbm * 128
output_block_bytes = 64 * element_size
```

`[N_hbm,128,1,D]` 的每个 token row 在最后一维上连续，所以可以安全地视为
`output_rows` 个定长 byte row。上述 `view` 不分配、不拷贝，只是为了与
当前 `NpuRuntime.batch_get` 的“第一维是 output rows”契约对齐。Adaptor
只计算 Runtime request 需要的相对 row，不做裸指针算术。

## 7. Decode Runtime read 详细时序

### 7.1 每层热路径

```text
Step 1  A5 完成
        topk_index/topk_slots/miss_count 就绪
        cache_slots 已经更新

Step 2  Adaptor device transform
        对所有 Bmax*2048 位置写固定 scratch
        活跃 miss → 正常 indices/output_indices, mask=1
        hit/PAD/padding → mask=0，indices/output_indices 写安全占位值 0

Step 3  KvafRuntimeClient 构造 NOPE kvaf::BatchGetRequest
        调用 Runtime::BatchGet
        Runtime 将结果原地写入
        resident_nope_cache.view(N_hbm*128,512)[physical_token_row]

Step 4  KvafRuntimeClient 构造 ROPE kvaf::BatchGetRequest
        调用 Runtime::BatchGet
        Runtime 将结果原地写入
        resident_rope_cache.view(N_hbm*128,64)[physical_token_row]

Step 5  RuntimeResultAdapter 检查两次 Runtime 提交结果和 caller stream 顺序
        SfaHandoffBuilder 生成 SFA 输入
        SFA-Offload
        sparse_indices = topk_slots
        根据 resident block table 读入 2048 个 sparse slot
        再根据 tail_info 读连续 dense tail
```

### 7.2 mask 的精确公式

```text
prefix_miss = (k < miss_count[b])
valid_p     = (topk_index[b,0,k] >= 0)
valid_s     = (topk_slots[b,0,k] >= 0)
active_row  = (b < active_B) and row_mode == SPARSE

mask[b,k] = uint8(prefix_miss && valid_p && valid_s && active_row)
```

正常运行时，A5 前缀契约保证有效 miss 都位于前缀。仍然保留
`p/s >= 0` 检查，可以使短序列和 PAD 更安全。

### 7.3 为什么 hit 不需要 Runtime lookup

A5 已经保证 hit 的 `topk_slots` 仍指向有效 resident KV。因此 hit 项的
`mask=0`，`Runtime::BatchGet` 不对该项产生数据搬运。SFA 仍会在后续按
`topk_slots` 读取它们。

### 7.4 stream 契约

MVP 必须使用同一 caller NPU stream：

```text
A5
  happens-before adaptor transform
  happens-before Runtime::BatchGet(NOPE)
  happens-before Runtime::BatchGet(ROPE)
  happens-before RuntimeResultAdapter / SfaHandoffBuilder
  happens-before SFA-Offload
```

`Runtime::BatchGet` 的 execution option 默认 `synchronize=false` 没有问题，
因为同 stream 的后续 SFA 天然等待
两个 UDMA READ 完成。如果未来将两个 plane 分到不同 stream，必须在 SFA
前为两条 stream 都建立 event wait。

## 8. 满块 Runtime write 详细时序

### 8.1 触发条件

对于当前 request，只有在某个 logical MLA block 的 128 个 token 都已经写入
当前 layer 的 HBM cache，且本层对该 token 的生产者已结束时，才可以触发。

在单 Token Decode 中，每个请求每步最多新满一个块。建议在当前 layer
SFA 之后的 post-layer/post-forward hook 中执行，不把 BatchPut 放入当前的
captured decode graph。

### 8.2 两 plane 提交

```text
full logical block L becomes ready
  → resolve vLLM source physical block id
  → Adaptor 构造 NOPE kvaf::BatchPutRequest
  → KvafRuntimeClient 调用 Runtime::BatchPut(NOPE)
  → Adaptor 构造 ROPE kvaf::BatchPutRequest
  → KvafRuntimeClient 调用 Runtime::BatchPut(ROPE)
  → if both success:
       remote_state[L] = COMMITTED
       schedule AccessTable epoch refresh
     else:
       remote_state[L] = NOT_VISIBLE
       do not expose L to A5/BatchGet
```

当前 `Runtime::BatchPut` 的一个 `BatchPutRequest` 只接受一段连续
input base。NOPE 和 ROPE 来自不同 tensor，因此 MVP 向 Runtime 提交
两个 `BatchPutRequest`。同一 plane 中的多个新满块可以合成
一个 batch，通过 `keys/input_offsets/lengths` 表达。

### 8.3 可见性和下一步规则

本轮刚满的块在 BatchPut/CommitPut 成功前不属于 remote candidate。为了避免
A5 更新 mapping 后却无法回载，设计采用严格的一阶段可见性：

```text
HBM block ready
  → Runtime::BatchPut data path success
  → Runtime 内部 CommitPut success
  → both planes present
  → 通过 Runtime 准备 AccessTable new epoch
  → next A5 invocation may select/miss this logical block
```

## 9. dense tail 和 A5 candidate 范围

A5 非 MTP ABI 没有 `tail_info`，也没有保护“必须常驻的未满块”的 row mode。
因此 MVP 不能让 A5 将未 remote-committed tail 当作普通可淘汰 candidate。

设计规则：

```text
remote_sparse_tokens
  = 从序列起点开始、NOPE/ROPE 都已 commit 的完整块范围

A5 actual_seq_lengths_key
  = remote_sparse_tokens（或由 VllmA5Bridge 提供等价 candidate 边界）

dense tail
  = 未进入这个 remote sparse prefix 的当前 HBM 连续区域
  → 通过 tail_info 直接交给 SFA-Offload
```

对于还没有进入 sparse decode 的短请求，vLLM 保留 DENSE 旁路，不调用
A5 + Adaptor + `Runtime::BatchGet`。从 DENSE 进入 SPARSE 的 first-fill 由
`VllmA5Bridge` 负责。

## 10. AccessTable 的生命周期和 graph 策略

### 10.1 为什么需要 epoch

KVAF 明确规定：新 `Runtime::BatchPut` 不会自动更新旧 AccessTable。
因此 Adaptor 要维护：

```text
AccessTableEpoch
├─ committed block key list（按固定顺序）
├─ NOPE AccessTableHandle
├─ ROPE AccessTableHandle
├─ stable NOPE DeviceAccessTable buffers
├─ stable ROPE DeviceAccessTable buffers
├─ remote_block_ordinal mapping
└─ last_use_event / stream ownership
```

### 10.2 epoch 切换

```text
1. 等待旧 epoch 的最后一次 `Runtime::BatchGet` 使用完成
2. `KvafRuntimeClient` 对新 committed key set 调用
   `Runtime::PrepareAccessTable`（图外）
3. 通过 Runtime/NPU Runtime facade 将新 projected view materialize 到 Adaptor
   持有的固定 device buffer；Adaptor 不解析 view 中的地址
4. 更新 remote_block_ordinal 和 valid 内容
5. 将控制面 handle 切换到新 epoch
6. 在旧 stream/event 完成后调用
   `Runtime::ReleaseAccessTable(old_handle)`
```

图回放只捕获 DeviceAccessTable、scratch 和 output tensor 的**地址**，回放前可以更新
其内容。因此目标实现需要 Runtime 支持向 caller-owned 固定 device table 重新
materialize；当前 Python `access_table_view(device="npu")` 每次创建新 tensor 的方式
不能直接用于长期 graph replay。

### 10.3 A5 入图前置条件

当前 A5 Torch wrapper 在每次调用中 `at::empty` 三个输出。目标 graph 路径需要
`npu_lightning_indexer_decode_update_a5_out`或等价 caller-owned output ABI：

```text
topk_index [Bmax,1,2048]
topk_slots [Bmax,1,2048]
miss_count [Bmax]
```

张量地址固定，本轮 `B` 只决定有效前缀/PAD mask。这是
`VllmA5Bridge` 的必要前置工作，不是 `KvafRuntimeClient` 可以规避的问题。

A5 公共 ABI 没有 `row_modes`。因此固定 `Bmax` graph 还必须在以下两种方案中
明确选一种：

- 使用 exact-B/bucketed graph，A5 只看真实活跃行；
- 在 A5 源码回归中证明 `actual_seq_lengths_key=0` 的 PAD 行完全 no-op，
  然后才允许把 `Bmax-B` 行一起捕获。

MVP eager 正确性路径先使用 exact `B`；在没有 PAD 行源码回归证据前，
不将“零长度行必然安全”当作已实现事实。

### 10.4 projection 表的规模风险

token projection 使 address/valid entry 数放大 128 倍。这对小规模 correctness MVP
可接受，但对多 layer、大 batch、128K context 的 full graph 可能占用过多 HBM。

因此在扩展到生产规模前，必须根据
`layers * committed_blocks * 128 * (8-byte address + 1-byte valid)` 做显式内存预算。
如果超过预算，应切换到方案 B：块级 AccessTable + device `source_offsets`。

## 11. 失败和一致性语义

### 11.1 `Runtime::BatchPut` 失败

KVAF Runtime 已提供 `PreparePut → Execute → Commit/Abort` 语义。Adaptor 的规则是：

- 任一 plane 失败，整个 MLA logical block 不可 remote-visible；
- 不更新 `remote_block_ordinal`；
- 不扩大 A5 candidate prefix；
- 已成功 plane 可以立即通过 `Runtime::BatchFree` 回收，或保留为 orphan
  由请求结束回收，
  但不得对 BatchGet 可见。

### 11.2 `Runtime::BatchGet` 失败为什么更危险

A5 在 BatchGet 之前已经原地更新 `cache_slots`。如果回载失败，映射已经声称
miss KV 存在于新 slot，但该 slot 仍是旧数据或部分数据。A5 没有输出旧
victim mapping，Adaptor 不能低成本回滚。

MVP 采用 fail-stop：

```text
Runtime::BatchGet failure
  → 禁止当前 layer 继续 SFA
  → 将 request/layer resident state 标记为 POISONED
  → 重新 first-fill/rebuild，或终止请求
  → 不尝试在缺少 victim journal 时原地回滚
```

### 11.3 当前 KVAF status 边界

当前 `Runtime::BatchGet` 返回 `Status`，但 enqueue-only 路径下的 AIV
`core_status` 是 Runtime/Operator 内部 buffer，不作为一个图内输出交给 Adaptor。
因此分两个阶段：

- correctness/eager bring-up：使用 `synchronize=true` 或显式 stream 同步检查，
  只在两 plane 成功后调 SFA；
- graph/steady-state POC：通过控制面预验证消除 invalid lookup，数据依赖使用
  同 stream；设备或 transport 异常按 worker fail-stop 处理。

如果要求“图内 Runtime BatchGet 失败必须阻止 SFA 读取”的强一致性，
需要在 Runtime 公共契约中新增
caller-visible device completion/status 和 SFA gate，或将 A5 mapping update 拆成 prepare/commit
两阶段。这不在 MVP 数据通路的实现范围内，但是生产化前必须解决的缺口。

### 11.4 输入安全检查的分层责任

对每个 `mask=1` 项，Adaptor 保证 vLLM 语义正确：

- `0 <= p < remote_sparse_tokens`；
- `remote_block_ordinal >= 0`；
- `0 <= s < resident_budget_tokens`；
- resident block table entry 有效；
- physical output row 位于 `N_hbm*128` 内；
- NOPE/ROPE dtype 和存储 layout 与 Key `layout_version` 一致。

Runtime 保证 KVAF 公共请求正确：

- AccessTableHandle 存在且未 release；
- `indices/mask/output_indices/output` 的 device、长度和 element size 符合契约；
- projected AccessTable valid entry 的 length 与 `output_block_bytes` 一致；
- NPU 路径具有与 handle 对应的 DeviceAccessTable。

地址范围、transport 和 output buffer 底层可访问性由 Runtime 继续向下交给
AccessOperator 检查，不回流为 Adaptor 的裸地址契约。

## 12. 请求和逐层生命周期

### 12.1 Request start

```text
分配 stable request_pool_row
初始化每层 A5 cache_slots state
初始化 remote block state/ordinal table
准备固定容量 adaptor scratch
绑定 request_generation_id
```

### 12.2 DENSE 阶段

```text
vLLM 保持完整/dense HBM 路径
满块可以提前通过 Adaptor 调用 Runtime::BatchPut
但不调用 A5 + Runtime::BatchGet
```

### 12.3 ENTER SPARSE

```text
VllmA5Bridge 建立 resident budget 和 first-fill
确认 remote committed prefix 无空洞
建立 tail_info
开始使用 A5 输出驱动 KVAF miss 回载
```

### 12.4 Steady sparse Decode

```text
每 layer：vLLM/A5 输出 → Adaptor → 2*Runtime::BatchGet
          → Adaptor SFA handoff → SFA → optional 2*Runtime::BatchPut
每 step：新 committed 块在下一 epoch 可见
```

### 12.5 Request finish

```text
等待相关 Runtime::BatchGet/BatchPut stream 完成
通过 Runtime::ReleaseAccessTable 释放 handles
通过 Runtime::BatchFree 回收 request-generation 所有 NOPE/ROPE keys
释放 request pool row
行再分配前清空每层 cache_slots/remote state
```

## 13. 具体数值例子：Adaptor 到底做了什么

真实 `K=2048`，下例只写前 4 个位置。

### 13.1 A5 输出

```text
B = 1
miss_count[0] = 2

topk_index[0,0,:4] = [257,513,8,130]
topk_slots[0,0,:4] = [  5,130,9,  2]
```

前两项是 miss，后两项是 hit。Adaptor 生成：

```text
mask[0,:4] = [1,1,0,0]
```

### 13.2 DDR source 翻译

假设 request stable pool row 为 `3`，当前 layer 的逻辑块映射：

```text
remote_block_ordinal[3,2] = 41
remote_block_ordinal[3,4] = 77
```

则：

```text
p=257:
  logical_block = 257//128 = 2
  token_offset  = 257%128  = 1
  indices       = 41*128+1 = 5249

p=513:
  logical_block = 513//128 = 4
  token_offset  = 513%128  = 1
  indices       = 77*128+1 = 9857
```

NOPE projected table 中：

```text
address[5249] = NOPE_block_41_base + 1 * 1024 bytes
address[9857] = NOPE_block_77_base + 1 * 1024 bytes
```

ROPE projected table 中：

```text
address[5249] = ROPE_block_41_base + 1 * 128 bytes
address[9857] = ROPE_block_77_base + 1 * 128 bytes
```

Adaptor 自身只生成 `5249/9857`；上面的地址派生由 KVAF Runtime 在
AccessTable materialize 阶段完成。

### 13.3 HBM destination 翻译

假设：

```text
resident_block_table[0,0] = 40
resident_block_table[0,1] = 44
```

则：

```text
s=5:
  logical resident block = 0
  offset                 = 5
  physical block         = 40
  output_indices         = 40*128+5 = 5125

s=130:
  logical resident block = 1
  offset                 = 2
  physical block         = 44
  output_indices         = 44*128+2 = 5634
```

### 13.4 Adaptor 生成的两个 Runtime 输入

```text
NOPE kvaf::BatchGetRequest
  indices        = [5249,9857,0,0,...]
  mask           = [1,1,0,0,...]
  output_indices = [5125,5634,0,0,...]
  output_block_bytes = 1024
  output         = resident_nope_cache.view(N_hbm*128,512)

ROPE kvaf::BatchGetRequest
  indices        = [5249,9857,0,0,...]
  mask           = [1,1,0,0,...]
  output_indices = [5125,5634,0,0,...]
  output_block_bytes = 128
  output         = resident_rope_cache.view(N_hbm*128,64)
```

`KvafRuntimeClient` 分别将两个 request 传入 `Runtime::BatchGet`。Adaptor 不传入
远端地址、transport descriptor 或 Operator kernel args。

### 13.5 Runtime 结果经 Adaptor 转为 SFA 输入

Runtime 两次调用返回 `Status`，并将 KV 原地写入上面的两个 output view。
当 Runtime Status 成功且 caller stream 顺序满足后，Adaptor 生成 SFA handoff：

```text
resident slot 5   已是 p=257 的 NOPE+ROPE
resident slot 130 已是 p=513 的 NOPE+ROPE
resident slot 9/2 本来就是 p=8/130，没有传输

SfaHandoff
  sparse_indices       = [5,130,9,2,...]
  tail_info            = [tail_slot_start,tail_token_count]
  resident_nope_cache  = Runtime 已原地写入的原 tensor
  resident_rope_cache  = Runtime 已原地写入的原 tensor
  resident_block_table = vLLM 当前 layer block table
```

这个例子体现了 Adaptor 的全部核心价值：**它把 A5 的业务索引
`(p,s,miss)` 转成 KVAF Runtime 公共请求，再把 Runtime 的 Status/原地 HBM
结果转成 SFA handoff。**

## 14. 接口草案（设计层，非实现）

### 14.1 Adaptor 上层接口

```text
Initialize(config, vllm_cache_layout, kvaf_runtime)
  -> AdaptorState

AcquireRequest(request_generation_id)
  -> RequestAdaptorState

PrepareLayerAccessEpoch(requests, layer_id, committed_blocks)
  -> LayerAccessEpoch

PrepareDecodeMisses(selection, decode_context)
  -> RuntimeReadPlan         # 只生成 Runtime 请求所需的元数据/scratch

LoadDecodeMisses(plan, stream)
  -> RuntimeReadResult       # 内部只调用 2*Runtime::BatchGet

BuildSfaHandoff(selection, runtime_read_result, decode_context)
  -> SfaHandoff

PutCompletedBlocks(block_jobs, layer_id, stream)
  -> RuntimeWriteResult      # 内部只调用 2*Runtime::BatchPut

ReleaseRequest(request_generation_id)
  -> Status
```

### 14.2 `KvafRuntimeClient` 与当前 Runtime 公共接口的精确对齐

```text
KvafRuntimeClient::PrepareAccessEpoch
  → Runtime::PrepareAccessTable(AccessTablePrepareRequest, &handle)
  → Runtime::GetAccessTableView(handle, &view)
  → NPU Runtime facade materializes Runtime-owned view

KvafRuntimeClient::LoadMisses
  → Runtime::BatchGet(nope_request, execution)
  → Runtime::BatchGet(rope_request, execution)
  → RuntimeReadResult{nope_submit_status,rope_submit_status,
                      in_place_outputs,caller_stream_order}

KvafRuntimeClient::StoreFullBlocks
  → Runtime::BatchPut(nope_request, execution)
  → Runtime::BatchPut(rope_request, execution)
  → RuntimeWriteResult{nope_status,rope_status}

KvafRuntimeClient::ReleaseAccessEpoch
  → Runtime::ReleaseAccessTable(handle)

KvafRuntimeClient::FreeRequestBlocks
  → Runtime::BatchFree(BatchFreeRequest{keys})
```

上述 client 不暴露 `AccessOperator::ExecuteBatchGet/Put`，不接收或返回
`BatchGetExecRequest/BatchPutExecRequest`，不接收 AIV core 数、SQ descriptor 或远端地址。

### 14.3 MVP 所需的最小 Runtime 公共扩展面

```text
AccessTablePrepareRequest
├─ flat_keys
└─ optional projection={entries_per_key,entry_bytes,entry_stride}

NPU Runtime facade: access_table_view_into(
    access_table_handle,
    caller_owned_addresses,
    caller_owned_valid,
)
```

第一项让 `Runtime::PrepareAccessTable/GetAccessTableView/BatchGet` 拥有一致的 token-entry
语义；第二项用于 graph 中固定 DeviceAccessTable 地址。两项都是 Runtime
公共能力，不允许 Adaptor 绕过 Runtime 自行派生地址。

## 15. 测试矩阵和验收标准

### 15.1 纯翻译单元测试

- `miss_count=0`：全部 mask=0，两个 Runtime read request 不搬数据；
- `miss_count=2048`：全部有效；
- 短序列：A5 `-1` padding 必须 mask=0；
- 块边界 `p=127/128/255/256`；
- resident 边界 `s=127/128`；
- 多个 miss 位于同一 remote block 的不同 token offset；
- 同一 source token 写入不同 request 的不同 physical block；
- batch condense/reorder 后通过 `request_pool_entries` 仍找到正确 ordinal。

### 15.2 Adaptor ↔ Runtime read 集成测试

- 测试只从 Adaptor 调用 `Runtime::BatchGet`，不直接 mock/call
  `AccessOperator::ExecuteBatchGet`；
- 两 plane 字节与原 HBM full block 指定 token 完全一致；
- Runtime 返回值为 Status，KV 结果体现为 caller-owned output view 原地变化；
- `[N_hbm,128,1,D] → [N_hbm*128,D]` 无拷贝 view 与 NPU Runtime
  output-row 契约一致；
- hit destination 原数据不变；
- miss destination 只改变目标 token row；
- invalid projected entry 在 `mask=0` 时 Runtime 返回成功且 output 不变；
- invalid entry 在 `mask=1` 时明确失败；
- 不同 output tensor A→B→A 的 local-buffer 注册复用；
- 同 stream A5→Runtime(NOPE)→Runtime(ROPE)→Adaptor handoff→SFA
  不需 Host 同步仍保持正确。

### 15.3 Adaptor ↔ Runtime write/生命周期测试

- 测试只从 Adaptor 调用 `Runtime::BatchPut/BatchFree`；
- 未满块不 Put；
- 刚满块仅 Put 一次；
- 任一 plane 失败时 logical block 不可见；
- 两 plane 成功后只在下一 epoch 可见；
- 旧 AccessTable 在最后一个 stream 完成前不 release；
- request finish 后 Key 可通过 `Runtime::BatchFree` 回收，pool row 复用不泄漏旧 mapping。

### 15.4 A5 → Adaptor → SFA 端到端

- 全 hit、小 miss、512/513 边界、全 2048 miss；
- A5 输出的 miss 前缀在 HBM 回载后逐项与 DDR 字节一致；
- `topk_slots == cache_slots_after.gather(topk_index)` 不变式仍成立；
- SFA output 与使用本地 DRAM/KSC 回载的对照路径在允许误差内一致；
- dense tail 不会生成远端 miss，SFA 仍读取完整 tail；
- NOPE/ROPE 任一 plane 未就绪时不调 SFA。
- SFA 入口不接收 Runtime request/handle/Key，只接收 resident cache/block table
  和 Adaptor 生成的 `sparse_indices/tail_info`。

### 15.5 graph 验收

- capture/replay 期间 A5 outputs、indices、mask、output_indices、DeviceAccessTable
  和 resident cache 的底层地址不变；
- 多次 replay 只修改张量内容，可得到不同 miss 结果；
- replay 热路径不做 Host Key 解析、Meta lookup、tensor allocation 或 D2H；
- BatchPut 明确位于 capture 外；
- 最坏 `Bmax*2048` miss 下，Adaptor 发出的固定容量 Runtime 请求可成功提交并完成；
  AIV core、transport SQ depth 和分片/多 launch 是 Runtime 向下实现的验收项，
  不进入 Adaptor 接口。

## 16. 实现前必须固定的契约

下列项不影响本文的主流程，但在编码前必须形成常量/接口定义：

1. `Bmax`、resident budget 和最大 remote block 数；
2. A5 如何通过 stable `req_pool_entries` 访问每请求状态；
3. A5 caller-owned out ABI 的最终 Torch/C++ 名称；
4. `tail_info` 的产生时点和逻辑 slot 空间；
5. `KVBlockKey` 的字节编码、namespace 和 generation 回收规则；
6. `AccessTablePrepareRequest.projection` 和 Runtime-owned projected view 的精确字段；
7. NPU Runtime 如何将 Runtime view materialize 到 caller-owned 固定 buffer；
8. worst-case miss 下 Runtime 如何向下选择 AIV core/SQ/launch 分片
   （Runtime 实现契约，不进入 Adaptor ABI）；
9. correctness 模式下 Runtime Status 和 stream 完成如何在 SFA 前被观测；
10. NOPE/ROPE 两个 Runtime write 部分成功时的 orphan 回收策略；
11. token-projected table 在目标模型/layer/batch/context 下的 HBM 预算上限；
12. A5 graph 采用 exact-B bucket，还是先增加并验证 PAD row 契约。

## 17. 最终设计决策汇总

| 问题 | MVP 决策 |
|---|---|
| A5 输入范围 | 单 Token Decode，64 Indexer heads，非 MTP |
| Adaptor 的核心输入 | vLLM/A5 暴露的 `topk_index/topk_slots/miss_count` + vLLM 寻址上下文 |
| Adaptor 对下唯一边界 | `KvafRuntimeClient → KVAF Runtime` 公共方法/request 类型 |
| Adaptor 禁止直接依赖 | MetaManager、AccessOperator、ExecRequest、AIV ABI、UBA 地址 |
| hit/miss 决策者 | A5 |
| mask | `k < miss_count[b]` 且 `p/s` 有效 |
| DDR 写入粒度 | 128-token full block，NOPE/ROPE 分 plane |
| DDR 读取粒度 | 1 token，NOPE/ROPE 两次 `Runtime::BatchGet` |
| MVP 块内寻址 | `AccessTablePrepareRequest.projection` + Runtime-owned token AccessTable |
| 生产优化候选 | Runtime `BatchGetRequest` 扩展 `source_offsets` + Runtime device size validation |
| HBM 目标 | vLLM resident paged cache 原地 token row |
| Runtime read 输出 | 当前为 `Status + caller stream 顺序 + request.output 原地 resident HBM`；无独立 completion handle |
| SFA 输入 | Runtime 已就绪 resident cache/block table + `topk_slots [B,1,2048] + tail_info [B,2]` |
| 中间 KV gather tensor | 无 |
| 顺序 | vLLM/A5 输出 → Adaptor → Runtime NOPE/ROPE read → Adaptor handoff → SFA，同 stream |
| Runtime write 图模式 | MVP 图外 synchronous eager |
| Runtime read 失败 | fail-stop/状态 poisoned，Adaptor 不生成可执行 SFA handoff，不尝试无 journal 回滚 |
| 未满 tail | 不进入 A5 remote candidate，由 `tail_info` 交 SFA |

## 18. 主要源码与设计依据

### KVAF

- `KVAF/docs/kvaf_architecture_design.md:28-67,149-166,368-457,954-1083`
- `KVAF/include/kvaf/requests.h`
- `KVAF/runtime/include/kvaf/runtime/runtime.h`
- `KVAF/runtime/src/runtime.cc`
- `KVAF/runtime/README.md`
- `KVAF/bindings/npu_python_runtime.cc`

Runtime 内部数据面限制的依据（不是 Adaptor 直接依赖）：

- `KVAF/access_operator/demo/aiv_batch_get/include/batch_get_abi.h`
- `KVAF/access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc`
- `KVAF/access_operator/demo/aiv_batch_get/README.md:18-25`

### vLLM v0.23

- `vllm_ascend/attention/sfa_v1.py:1545-1755`
- `vllm_ascend/dsa_offload/contracts.py`
- `vllm_ascend/dsa_offload/ops.py`
- `vllm_ascend/dsa_offload/resident_pool.py`
- `vllm_ascend/dsa_offload/runtime.py:82-110,679-805`
- `vllm_ascend/dsa_offload/request_cache_layout.py`

### A5

- 见 [LIDU-A5算子实现报告](./LIDU-A5算子实现报告.md) 第 11 节的源码索引。
