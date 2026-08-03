# 当前 GLM 分支：单层 Decode 的完整 KV 数据流

> 本文只看当前 `vllm-ascend-v0.19.1rc1-gs-glm`（代码目录 `vllm-glm`）在一个 attention layer 内的 KV 链路：当前 token 生成并写入 Indexer/MLA cache → vLLM 请求和 block 信息变成 device metadata → LIDU 判断 hit/miss → KSC 把 miss KV 从 DRAM 换入 resident HBM → SFA-Offload 读取并返回 attention 结果。
>
> 本文的 `p` 都是“请求内的 original token position”，不是词表 token ID。数值例子是为说明公式而构造，不是从真实请求采样的值。

## 0. 先给结论

当前 GLM 分支有两条彼此独立、只在算子中通过 index 会合的数据线：

```text
KV payload 线（真正的 BF16/FP16 数据）

hidden_states
├─生成 Indexer K [T,1,128]
│  └─写入完整 Indexer HBM cache [N_indexer,128,1,128]
│
└─生成 MLA NOPE_K / ROPE_K
   └─写入 resident HBM cache
      ├─ NOPE_K [N_hbm,128,1,512]
      └─ ROPE_K [N_hbm,128,1,64]
             │
             ├─新满 block：整 block HBM → swapped DRAM
             └─miss token：单 token DRAM → resident HBM

metadata/index 线（主要是 INT32）

当前 layer 的 LIDU 在一次算子调用内完成：
完整 Indexer K 打分并选出 original position p
→ 查询 cache_slots[pool_row,p]，命中则取得 s，未命中则分配 s
→ 同时写入两个固定容量输出：
     topk_index : INT32 [B,1,16384]，在有效前缀保存 p
     topk_slots : INT32 [B,1,16384]，在有效前缀保存与 p 对应的 resident slot s

其中 16384 只是固定 buffer 容量；真正有效的范围由消费者决定：
├─ KSC 只读取 [0, miss_count[b]) 的 (p,s)，即本轮 miss 搬运任务
└─ SFA-Offload 只读取 topk_slots[b,0,0:2048]，即本轮 Top-K 的 2048 个 s

steady sparse Decode 中，前 2048 项是同一批 Top-K 的 (p,s)，并已把 miss 排在前缀；
首次填充 resident cache 时，miss_count 可以是 6144/10240/12288，但 SFA 仍只计算前 2048 项。

完整 cache_slots 存储为 [num_layers,pool_rows,W]；执行当前 layer 时，LIDU 只接收
该层的 [pool_rows,W] view，并通过 req_pool_entries[b] 把本轮 batch row b 映射到 pool_row。
```

最重要的边界是：

- KV payload 没有被“转成 H2D metadata”。H2D 传的是请求行、block ID、行模式等寻址数据；KV payload 始终保持 `[512]` NOPE 和 `[64]` ROPE 的内容。
- 完整 Indexer K 一直在 HBM，用来对完整历史打分；被卸载和换入的是 MLA NOPE/ROPE KV，不是 Indexer K。
- `p` 只在“当前请求”内唯一。要唯一定位一份 KV，语义上需要 `(request/cache namespace, layer_id, p, KV plane)`。
- 当前集成是 single-token Decode-only：`T=B`。通用算子的 shape 能力不等于当前 vLLM 路径已接通 MTP。

源码依据：`vllm-glm/vllm_ascend/attention/sfa_v1.py:1066-1245,1317-1555`，`vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:480-558`；single-token 集成限制见 `vllm-glm/vllm_ascend/worker/model_runner_v1.py:838-852`。

## 1. 符号与三类“位置”

| 符号 | 含义 | 当前值/范围 |
|---|---|---|
| `B` | 本轮 Decode 请求行数 | 动态 |
| `T` | TND layout 压平后的 query token 数 | 当前 single-token Decode 中 `T=B` |
| `N_idx` | Indexer query head 数 | LIDU ABI 支持 `32` 或 `64`；当前模型实例由 `indexer.n_head` 决定 |
| `N1` | 本 rank 的 SFA query head 数 | 配置相关 |
| `BS` | KV page/block 的 token 数 | `128` |
| `K` | SFA 真正计算的 sparse Top-K | `2048` |
| `C` | LIDU/KSC 物理输出容量 | `16384` |
| `p` | 请求内 original token position | `0 <= p < seq_len` |
| `s` | 请求的 resident logical slot | `0 <= s < resident budget + tail capacity` |
| `b` | 请求在当前 model forward 中的行 | `0..B-1`，可随 continuous batching 变化 |
| `r` | 请求持久的 resident pool row | 请求生命期内稳定 |

不要混淆下面三类整数：

```text
p = original position       例如 257，用来找原始历史 token
s = resident logical slot   例如 33，用来找当前 HBM 中的驻留位置
physical_block_id           例如 37/91/7，是某个 cache arena 中的 block 号
```

`physical_block_id` 不是裸指针。算子内部将它与 cache tensor 的 `base_addr` 结合：

```text
logical_block(x) = x // 128
offset(x)        = x % 128
physical_block  = block_table[b, logical_block(x)]
physical_row    = physical_block * 128 + offset(x)

NOPE address = NOPE_base + physical_row * 512 * sizeof(dtype)
ROPE address = ROPE_base + physical_row *  64 * sizeof(dtype)
```

所以 `base_addr` 是“当前 layer、当前 cache plane 的起始指针”。layer 不需要出现在 block table 的 shape 中，因为 Python 在调用算子前已经选定当前 layer 的 tensor/base pointer。

## 2. 真正的 KV payload：三张每层 cache plane

### 2.1 分配后的实际 shape

| payload | dtype | 每层实际 shape | 内存 | 生命期 | 用途 |
|---|---|---|---|---|---|
| 完整 Indexer K | BF16/FP16 | `[N_indexer,128,1,128]` | HBM | worker 生命期 | LIDU 对完整历史计分 |
| resident NOPE/latent KV | BF16/FP16 | `[N_hbm,128,1,512]` | HBM | worker 生命期 | KSC 写，SFA-Offload 读 |
| resident ROPE_K | BF16/FP16 | `[N_hbm,128,1,64]` | HBM | worker 生命期 | KSC 写，SFA-Offload 读 |
| DRAM NOPE arena | BF16/FP16 | `[N_dram,128,1,512]` | NPU-addressable swapped DRAM | worker 生命期 | full-block dump 写，KSC 读 |
| DRAM ROPE arena | BF16/FP16 | `[N_dram,128,1,64]` | NPU-addressable swapped DRAM | worker 生命期 | full-block dump 写，KSC 读 |

shape 中没有 `B` 和 layer 维：

- `N_*` 是当前 layer 的 physical block pool 容量，所有请求共享这个 pool；请求由 `block_table[b,:]` 隔离。
- 每个 layer 持有自己的三张 HBM tensor 和两张 DRAM arena；同一 physical block ID 在不同 layer 中因 base pointer 不同而指向不同 payload。
- 当前 split-cache 分配器为 Indexer 和 MLA 建立不同 KV group；默认 block 数比例是 Indexer:MLA=`3:1`，但这是容量配置，不改变单 block 布局。

分配依据：

- 通用 cache shape：`vllm-glm/vllm_ascend/attention/sfa_v1.py:118-125`。
- Indexer spec 明确 `num_kv_heads=1` 且 `head_size=self.head_dim`：`vllm-glm/vllm_ascend/patch/dsa_sparse/patch_deepseek_v2.py:27-37`。
- Indexer tensor view：`vllm-glm/vllm_ascend/worker/model_runner_v1.py:3895-3912`。
- MLA 按 `kv_lora_rank`/`qk_rope_head_dim` 拆成 NOPE/ROPE shape：`vllm-glm/vllm_ascend/worker/model_runner_v1.py:3687-3703,3991-4025`。
- DRAM arena 直接沿用 HBM 单 block shape：`vllm-glm/vllm_ascend/dsa_sparse/dsa_ascend_hot_kv_store.py:50-70,128-155`。

### 2.2 当前 token 如何产生并写入三张 cache

对当前 single-token Decode，`T=B`：

| 阶段数据 | dtype | shape | 产生者 → 消费者 | 具体含义 |
|---|---|---|---|---|
| `kv_no_split` | BF16/FP16 | `[T, kv_lora_rank+qk_rope_head_dim]`，GLM 当前是 `[T,576]` | fused QKV-A projection → `npu_kv_rmsnorm_rope_cache` | 当前 token 的 latent KV + RoPE K |
| `kv_no_split` operator view | BF16/FP16 | `[T,1,1,576]` | `exec_kv` → cache-write op | `N=1,S=1`，不是历史 sequence 维 |
| `k_li` | BF16/FP16 | `[T,1,128]` | Indexer K projection/RMSNorm/RoPE → scatter | 当前 token 的 Indexer K |
| `indexer_slot_mapping` | INT32 | `[T]` | Indexer block table + original positions → scatter | Indexer HBM cache 的扁平 token row |
| `slot_mapping` | INT32 | `[T]` | resident HBM block table + resident positions → MLA write | resident HBM cache 的扁平 token row |

流程是：

```text
hidden_states [T,hidden]
│
├─ wk/k_norm/RoPE
│    → k_li [T,1,128]
│    → scatter_nd(indexer_slot_mapping)
│    → Indexer K cache [N_indexer,128,1,128]
│
└─ fused_qkv_a_proj
     → kv_no_split [T,576]
     → view [T,1,1,576]
     → RMSNorm + split + RoPE
     → 按 slot_mapping 原址写入
        NOPE [N_hbm,128,1,512]
        ROPE [N_hbm,128,1,64]
```

同一个当前 token 会有两个不同的 cache-write slot：

- Indexer plane 使用 original position，保留完整密集历史。
- MLA plane 对 sparse request 使用 resident position，只保留 resident budget 与 dense tail。

源码依据：`vllm-glm/vllm_ascend/attention/sfa_v1.py:884-926,1022-1064,1409-1425,1497-1516`。

## 3. vLLM 的“KV key”如何变成算子 metadata

### 3.1 没有一个独立的“全局 token key”

vLLM 调度/缓存管理侧先以 `request_id` 区分请求，每个请求保存多个 KV cache group 的 physical `block_ids`。到 worker 侧后：

```text
request_id
→ CachedRequestState.block_ids[group_id]
→ InputBatch 当前行 b
→ block_table_cpu[b, logical_block] = physical_block_id
→ block_table_device[B,M]
```

然后 token position `p` 只负责选择这一行中的列：

```text
logical_block = p // 128
offset        = p % 128
physical      = block_table_device[b, logical_block]
```

因此：

```text
请求身份：request_id → 本轮 batch row b
层身份：Python 已选定 layer tensor/base pointer
token 身份：请求内 original position p
plane 身份：Indexer / NOPE / ROPE 使用不同 tensor/base pointer
```

如果这里说的“vLLM KV key”特指 prefix-cache `block_hash`，它又是另一种东西：它标识一个完整 logical block 的内容，只在 CPU control plane 中用于 DRAM block 去重、分配和 refcount，不是 LIDU 输出的 `p`，也不直接 H2D：

```text
(request_id, pool row r, original logical block, block_hash)
→ DRAM store 在 CPU 上查找/分配 physical DRAM block
→ dram_block_table[r, original logical block] = physical DRAM block
→ 只把已整数化的 dram_block_table active rows H2D
→ KSC 只看 INT32 table，不看 hash
```

源码依据：

- scheduler 输出的 `block_ids` 进入 `CachedRequestState`：`vllm-glm/vllm_ascend/dsa_sparse/dsa_model_runner_adapter.py:154-194`。
- cached request 的新 block ID 更新 InputBatch block table：`vllm-glm/vllm_ascend/dsa_sparse/dsa_model_runner_adapter.py:234-342`。
- `BlockTable.append_row` 把 block ID 写入请求行：`vllm-glm/vllm_ascend/worker/block_table.py:93-110`。
- block hash 随 scheduler output 进 worker：`vllm-glm/vllm_ascend/patch/dsa_sparse/patch_scheduler_output.py:15-46`；DRAM store 用 hash 解决物理块并写逻辑表：`vllm-glm/vllm_ascend/dsa_sparse/dsa_hot_kv_store_core.py:498-515,578-690`。

### 3.2 Host → Device 和 device-only metadata 总表

| metadata | dtype/shape（active view） | CPU/Host 来源 | 如何到 device | device 消费者 | 生命期 |
|---|---|---|---|---|---|
| Indexer `block_table` | INT32 `[B,M_idx]` | `CachedRequestState.block_ids[indexer_group]` | `BlockTable.commit_block_table()` H2D | Indexer cache write、LIDU | 持久 buffer，行变更时刷新 |
| MLA/resident native `block_table` | INT32 `[B,M_hbm]` | `block_ids[full_group]` | 同上 | MLA cache write、SFA-Offload | 持久 buffer，行变更时刷新 |
| `query_start_loc` | INT32 `[B+1]` | 每请求 query token 数的累加 | `copy_to_gpu()` | slot-mapping kernel、TND 算子 | 每 forward |
| `positions` | INT64 `[T]` | `num_computed_tokens + query_pos` 的语义 | 当前非 PCP 路径在 device 直接计算 | Indexer slot mapping/RoPE | 每 forward |
| `resident_positions` | INT64 `[T]` | 先复制 original positions，sparse 行改成 resident position | 与 resident valid lengths 合并成一块 pinned slab H2D | MLA slot mapping/RoPE metadata | 每 forward |
| Indexer `slot_mapping` | INT32 `[T]` | 无成品 CPU tensor | device kernel：Indexer table + original positions | Indexer K scatter | 每 forward |
| MLA `slot_mapping` | INT32 `[T]` | 无成品 CPU tensor | device kernel：resident table + resident positions | MLA NOPE/ROPE cache write | 每 forward |
| `actual_seq_lengths_key` | INT32 `[B]` | computed/scheduled token 数 | ModelRunner 维护 device seq-len buffer | LIDU | 每 forward |
| `req_pool_entries` | INT32 `[B]` | `request_id → resident pool row r` | row-mode packed slab H2D | LIDU 用 `r` 选 `cache_slots` 行 | 请求绑定持久，本轮序列化 |
| `row_modes` | INT32 `[B]` | scheduler/request stage 投影 | 同一 row-mode slab H2D | LIDU：PAD/DENSE/SPARSE | 每 forward |
| KSC `hbm_block_table` | INT32 `[B,M_hbm_owner]` | native MLA table 的 active rows | 拷入 row-mode packed slab后 H2D | KSC destination translation | 每 forward view |
| KSC `dram_block_table` | INT32 `[B,M_dram_owner]` | DRAM store 中按 pool row `r` 维护的逻辑表 | active rows 刷到 device persistent buffer | KSC source translation | 表版本/请求生命期 |
| `cache_slots` | INT32 `[pool,W]` per layer | 不是每步 Host 数据 | 初始化时直接分配在 NPU，LIDU 原地更新 | LIDU | 请求×layer 跨 step 持久 |
| LIDU 四个输出 | INT32，见第 5 节 | 无 Host 输入值 | LIDU 在 device 直接写 caller-owned buffer | KSC/SFA-Offload | 每 layer、每 forward |

关键区分：`slot_mapping` 并非每步从 CPU 传一个“物理地址数组”；当前非 PCP 路径用 device 上的 `positions + block_table` 现场生成扁平 slot。

源码依据：

- block table CPU/GPU 双 buffer 与 H2D：`vllm-glm/vllm_ascend/worker/block_table.py:75-89,241-264`。
- device slot-mapping kernel 调用：`vllm-glm/vllm_ascend/worker/block_table.py:148-170`。
- split group 分别使用 original/resident positions：`vllm-glm/vllm_ascend/worker/model_runner_v1.py:1193-1213`。
- resident metadata 合并 H2D slab：`vllm-glm/vllm_ascend/worker/model_runner_v1.py:378-430,745-764`。
- row-mode slab、HBM table 与 DRAM table staging：`vllm-glm/vllm_ascend/dsa_sparse/dsa_row_mode_runtime.py:93-151,539-679,681-746,873-881`。

### 3.3 `b` 与 `pool row r` 的区别

```text
b = 当前 forward 的 batch row
    由 InputBatch 当前排序决定，请求 condense/reorder 后可变

r = resident pool row
    request_id 首次 acquire 时分配，请求生命期内稳定

req_pool_entries[b] = r
cache_slots_for_layer[r,p] = s
```

所以 LIDU 先用 `b` 读 `req_pool_entries[b]`，再用得到的 `r` 选中持久 `cache_slots` 状态行。这是为了让 batch 行可变，但请求的 hit/miss 状态不随之丢失。

源码依据：`vllm-glm/vllm_ascend/dsa_sparse/dsa_resident_pool.py:86-93,101-151`，`vllm-glm/vllm_ascend/dsa_sparse/dsa_input_batch_state.py:271-347,422-457`。

## 4. 卸载：新满的 MLA block 如何进入 DRAM

### 4.1 目的地址是如何建立的

DRAM store 在 worker 初始化时为每层预分配两个固定 arena，并在 store 级维护：

```text
(request_id, resident pool row r, original logical block)
→ DRAM physical block id
```

物理 DRAM block ID 在所有 layer 和 NOPE/ROPE plane 中共用同一编号语义；具体读哪一层、哪个 plane，由传入算子的 arena base pointer 决定。

例如，原始 logical block 2 在某次写满时位于 HBM physical block 44，DRAM allocator 为它预留 physical block 91：

```text
src_hbm_block_ids = [44]
dst_dram_block_ids = [91]

NOPE: HBM_NOPE[44,:,:] → DRAM_NOPE[91,:,:]
ROPE: HBM_ROPE[44,:,:] → DRAM_ROPE[91,:,:]
```

### 4.2 搬运时点、粒度与 shape

full-block dump 在当前 layer 的 MLA/SFA 成功结束后调用，每个 copy row 同时复制：

```text
NOPE: [128,512] = 128 * 512 * 2 bytes = 128 KiB
ROPE: [128, 64] = 128 *  64 * 2 bytes =  16 KiB
合计: 144 KiB / MLA block / layer
```

| full-block dump 参数 | dtype | 算子 view shape | 含义 |
|---|---|---|---|
| HBM NOPE source | BF16/FP16 | `[N_hbm,128,512]` | 当前 layer 的源 arena |
| HBM ROPE source | BF16/FP16 | `[N_hbm,128,64]` | 当前 layer 的源 arena |
| DRAM NOPE destination | BF16/FP16 | `[N_dram,128,512]` | 当前 layer 的目的 arena |
| DRAM ROPE destination | BF16/FP16 | `[N_dram,128,64]` | 当前 layer 的目的 arena |
| `src_block_ids` | INT32 | `[J]` | 本次真实 copy job 的 HBM physical block IDs |
| `dst_block_ids` | INT32 | `[J]` | 对齐的 DRAM physical block IDs；graph 中 `-1` 表示 no-op |

对外 ABI 使用 rank-3 view `[N,128,512/64]`；原 HBM/DRAM tensor 是 `[N,128,1,512/64]`，adapter 只 `squeeze(2)` 建 view，不复制 payload。kernel 内部把每个完整 block 按两个 plane 再分成连续 DMA chunk，分配给多个 AIV core；不是 Python 每生成一个 token 就往 DRAM 写一次。

时序：

```text
当前 layer 写入 MLA cache
→ LIDU → KSC → SFA-Offload
→ attention_finished(layer)
→ KvCacheFullBlockDump（如本轮有新满 block）
→ 下一个 forward 该 block 才进入 LIDU candidate/KSC 取回链路
```

当前 cache write、dump 与后续 forward 的 KSC 都提交到同一 NPU stream，依赖由 stream ordering 保证。

源码依据：

- 原 block ID 与 DRAM 目的 ID 的 model-forward 预留：`vllm-glm/vllm_ascend/dsa_sparse/dsa_forward_batch_builder.py:31-102`。
- layer 后置时点：`vllm-glm/vllm_ascend/ops/mla.py:196-219`，`vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:583-648`。
- dump ABI：`vllm-glm/csrc/kv_cache_full_block_dump/kv_cache_full_block_dump_torch_adpt.h:11-80`。
- 多 core 分 chunk 搬运：`vllm-glm/csrc/kv_cache_full_block_dump/op_kernel/kv_cache_full_block_dump.cpp:54-105,125-156`。

## 5. LIDU：用完整 Indexer K 选 p，再把 p 变成 resident slot s

### 5.1 LIDU 真实算子接口

| 参数 | dtype | 真实 ABI shape | 方向 | 值的含义 |
|---|---|---|---|---|
| `query` / `q_li` | BF16/FP16 | `[B,N_idx,128]` | 输入 | 当前 layer 、当前 query token 的 Indexer Q |
| `key` | BF16/FP16 | `[N_indexer,128,1,128]` | 输入 | 当前 layer 完整历史 Indexer K |
| `weights` | BF16/FP16 | `[B,N_idx]` | 输入 | 汇总 Indexer heads 得分的权重 |
| `req_pool_entries` | INT32 | `[B]` | 输入 | `b → r`，选中当前请求的持久 cache row |
| `cache_slots` | INT32 | `[pool,W]` | 可变输入 | `cache_slots[r,p]=s/-1`；最后一列保存 resident budget metadata |
| `row_modes` | INT32 | `[B]` | 输入 | `0=PAD,1=DENSE,2=SPARSE` |
| `actual_seq_lengths_key` | INT32 | `[B]` | 输入 | 每行完整 Indexer 有效历史长度 |
| Indexer `block_table` | INT32 | `[B,M_idx]` | 输入 | original logical block → Indexer HBM physical block |
| `topk_index` | INT32 | `[B,1,16384]` | 输出 | original positions `p`；miss pairs 位于前缀 |
| `topk_slots` | INT32 | `[B,1,16384]` | 输出 | 与 `p` 对齐的 resident slots `s` |
| `miss_count` | INT32 | `[B]` | 输出 | KSC 只处理每行 `[0,miss_count[b])` |
| `tail_info` | INT32 | `[B,2]` | 输出 | `[tail_slot_start, tail_token_count]` |

LIDU 用 Indexer block table 读 Key 并完成 Top-K；但它不读 NOPE/ROPE KV，也不搬运 KV payload。

源码依据：`vllm-glm/csrc/lightning_indexer_decode_update/lightning_indexer_decode_update_torch_adpt.h:12-134`。

### 5.2 `cache_slots` 如何判断 hit/miss

`cache_slots` 不在 LIDU 算子内部私有，而是 `DSAResidentTokenPool` 预先分配的持久 NPU tensor：

```text
_cache_slots [num_layers, pool, W]

执行 layer L 时：
cache_slots = _cache_slots[L]       # [pool,W]
r = req_pool_entries[b]
s = cache_slots[r,p]
```

```text
LIDU 选出 original position p
│
├─ cache_slots[r,p] = s >= 0
│    → hit
│    → topk_slots 写 s
│    → 不产生 KSC copy pair
│
└─ cache_slots[r,p] = -1
     → miss
     → 找一个不在本轮 Top-K 中的 old_p
     → 复用 old_p 的 resident slot s
     → 原地写：
          cache_slots[r,old_p] = -1
          cache_slots[r,p]     = s
     → 输出 miss pair (p,s)
```

更新顺序是：

```text
LIDU 先决定并写入 p → s 映射
→ KSC 根据 (p,s) 把真实 KV 填入 s
→ SFA-Offload 使用 s
```

因此 LIDU 结束的瞬间，映射已更新，但 miss payload 要到 KSC 结束后才存在于 s。三个算子在同一条顺序调用链/当前 stream 上，其他算子不会在 KSC 前使用这个新 s。

内核依据：

- 用 `req_pool_entries[b]` 选 cache row：`vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_kernel.h:216-239`。
- 在 `cache_slots[rowBase+p]` 中查 slot，选出 miss：`vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_vector.h:528-553`。
- 淘汰 old mapping、写入新 `p→s`：`vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_vector.h:568-649`。

### 5.3 为什么输出容量是 16384，SFA 却只算 2048

`16384` 是 LIDU/KSC 的物理 buffer 容量，不是每次 attention 要算 16384 个 sparse token：

- steady sparse Decode：前 `2048` 是 SFA sparse 选择，未使用尾部填 `-1`。
- 首次进入 sparse/first-fill：LIDU 可以一次输出并由 KSC 填充整个 resident budget，当前支持 `6144/10240/12288`，因此 buffer 需大于 2048。
- SFA-Offload 的 sparse compute count 始终是 `2048`；密集 tail 由 `tail_info` 另外表达。

源码依据：`vllm-glm/vllm_ascend/dsa_sparse/dsa_types.py:24-28`，`vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_common.h:45-54`，`vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_vector.h:750-848`。

## 6. KSC：把 miss pair `(p,s)` 变成两次物理寻址

### 6.1 KSC 真实算子接口

| 参数 | dtype | 真实 ABI shape | 作用 |
|---|---|---|---|
| HBM ROPE | BF16/FP16 | `[N_hbm,128,64]` | miss 目的 plane |
| HBM NOPE | BF16/FP16 | `[N_hbm,128,512]` | miss 目的 plane |
| DRAM ROPE | BF16/FP16 | `[N_dram,128,64]` | miss 源 plane |
| DRAM NOPE | BF16/FP16 | `[N_dram,128,512]` | miss 源 plane |
| `hbm_block_table` | INT32 | `[B,M_hbm_owner]` | resident logical block → HBM physical block |
| `dram_block_table` | INT32 | `[B,M_dram_owner]` | original logical block → DRAM physical block |
| `src_token_ids` | INT32 | `[B,1,16384]` | LIDU `topk_index`，original `p` |
| `dst_slots` | INT32 | `[B,1,16384]` | LIDU `topk_slots`，resident `s` |
| `copy_counts` | INT32 | `[B]` | LIDU `miss_count` |

源码依据：`vllm-glm/csrc/kvcache_scatter_copy/kvcache_scatter_copy_torch_adpt.h:12-94`。

### 6.2 每个 miss token 的地址公式

KSC 对每行只处理 `[0,miss_count[b])`，每个 active pair 做：

```text
源 p：
src_logical_block = p // 128
src_offset        = p % 128
src_physical      = dram_block_table[b, src_logical_block]

目的 s：
dst_logical_block = s // 128
dst_offset        = s % 128
dst_physical      = hbm_block_table[b, dst_logical_block]

DRAM NOPE[src_physical,src_offset,:512]
    → HBM NOPE[dst_physical,dst_offset,:512]

DRAM ROPE[src_physical,src_offset,:64]
    → HBM ROPE[dst_physical,dst_offset,:64]
```

这里的搬运粒度是单 token，即每个 miss 复制 `512+64` 个 BF16/FP16 元素，共 `1152 bytes`。

KSC 不在 Python 层对 2048 个 token 逐个发起 copy；一次算子调用收到整个 `[B,1,16384]` pair buffer 和 `[B]` count，kernel 把 `B*C` 展平为 work slots，跳过 count 之后的尾部，由多 core 并行处理离散 miss。所以从上层看是“把所有要读的 token 一次交给算子”，不需要感知一个名为 BatchGet 的独立接口。

内核依据：`vllm-glm/csrc/kvcache_scatter_copy/op_kernel/kvcache_scatter_copy_kernel.h:52-78,91-140`。

## 7. SFA-Offload：用 resident slot 读 HBM KV

### 7.1 SFA-Offload 真实接口

| 参数 | dtype | shape | 值的含义 |
|---|---|---|---|
| `query=ql_nope` | BF16/FP16 | `[B,N1,512]` | MLA latent query |
| `query_rope=q_pe` | BF16/FP16 | `[B,N1,64]` | query RoPE |
| `key=value` | BF16/FP16 | `[N_hbm,128,1,512]` | resident NOPE/latent KV；key/value alias 同一 tensor |
| `key_rope` | BF16/FP16 | `[N_hbm,128,1,64]` | resident ROPE_K |
| `sparse_indices` | INT32 | `[B,1,16384]` | LIDU `topk_slots`；稀疏计算使用前 2048 个 `s` |
| `tail_info` | INT32 | `[B,2]` | dense tail 的 resident slot 起点与长度 |
| resident `block_table` | INT32 | `[B,M_hbm]` | resident logical block → HBM physical block |
| `actual_seq_lengths_query` | INT32 | `[B]` | TND 的累积 query 边界；single-token 时类似 `[1,2,...,B]` |
| `actual_seq_lengths_kv` | INT32 | `[B]` | original/indexer 密集上下文长度 |
| output | BF16/FP16 | `[B,N1,512]` | SFA 的 latent attention 输出 |

SFA 不再看 original `p`，它看的是 KSC 已经物化好的 resident `s`：

```text
s
→ logical_block=s//128, offset=s%128
→ physical_hbm_block=block_table[b,logical_block]
→ DataCopyPA 从当前 layer 的 NOPE/ROPE base pointer 读 token row
→ QK + softmax + V
```

SFA 内部也不先在 Python 中把 2048 个离散 token gather 成新 KV tensor。kernel 按 sparse index 分 tile，在 `DataCopyPA` 内完成 page translation 和 GM→L1 读取，然后直接进入 matmul/softmax/value 流水。

源码依据：

- Python 交接：`vllm-glm/vllm_ascend/attention/sfa_v1.py:1204-1245`，`vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:545-565`。
- SFA-Offload ABI 与输出 shape：`vllm-glm/csrc/sparse_flash_attention_for_offload/sparse_flash_attention_for_offload_torch_adpt.h:19-107`。
- `DataCopyPA` 的 block-table 查表：`vllm-glm/csrc/sparse_flash_attention_for_offload/op_kernel/sparse_flash_attention_for_offload_service_cube_mla.h:61-100`。
- sparse slot 取值：`vllm-glm/csrc/sparse_flash_attention_for_offload/op_kernel/sparse_flash_attention_for_offload_service_cube_mla.h:523-538`。

## 8. 一个完整例子：Request A / layer 10 / `p=257`

以下数值是构造例子：

```text
本轮 batch row                  b = 1
Request A 的 resident pool row      r = 5
当前 attention layer             L = 10
LIDU 选中 original position       p = 257
分配的 resident logical slot  s = 33

Indexer block_table[1,2] = 37
DRAM    block_table[1,2] = 91
HBM     block_table[1,0] = 7
```

### 8.1 LIDU 如何读到 `p=257` 的 Indexer K

```text
p // 128 = 2
p %  128 = 1

Indexer physical block = indexer_block_table[1,2] = 37
Indexer physical row   = 37*128+1 = 4737

读取：Indexer_K_layer10_base + 4737*128*sizeof(dtype)
内容：128 个 BF16/FP16 元素
```

LIDU 对完整历史打分后选中 `p=257`。

### 8.2 `cache_slots` 判定 miss

```text
r = req_pool_entries[1] = 5
cache_slots_layer10[5,257] = -1

→ miss
→ 选定复用 s=33
→ cache_slots_layer10[5,old_p] = -1
→ cache_slots_layer10[5,257]   = 33

LIDU 输出前缀：
topk_index[1,0,0] = 257
topk_slots[1,0,0] = 33
miss_count[1]      >= 1
```

### 8.3 KSC 如何从 DRAM 91 搬到 HBM 7

```text
源 p=257：
src logical block = 257//128 = 2
src offset        = 257%128  = 1
src physical      = dram_block_table[1,2] = 91
src physical row  = 91*128+1 = 11649

目的 s=33：
dst logical block = 33//128 = 0
dst offset        = 33%128  = 33
dst physical      = hbm_block_table[1,0] = 7
dst physical row  = 7*128+33 = 929
```

KSC 在一次 token copy 中同时处理：

```text
DRAM_NOPE_layer10[91,1,:512] → HBM_NOPE_layer10[7,33,:512]
DRAM_ROPE_layer10[91,1,:64] → HBM_ROPE_layer10[7,33,:64]
```

### 8.4 SFA-Offload 如何再读到这个 KV

```text
sparse_indices[1,0,k] = 33

logical block = 33//128 = 0
offset        = 33%128  = 33
physical      = resident_block_table[1,0] = 7

NOPE element offset = (7*128+33)*512 = 475648
ROPE element offset = (7*128+33)* 64 =  59456
```

BF16/FP16 都是 2 bytes，因此当前 layer 内的字节地址是：

```text
NOPE_layer10_base + 951296 bytes，连续读 512 个元素
ROPE_layer10_base + 118912 bytes，连续读  64 个元素
```

这个例子展示了三套 table 不可互换：

```text
Indexer table : p → 完整 Indexer K 的 HBM 地址，用于选择
DRAM table    : p → 已卸载 MLA KV 的 DRAM 地址，用于 miss source
resident table: s → resident MLA KV 的 HBM 地址，用于 KSC destination 和 SFA source
```

## 9. 整条关键路径的时序

```text
当前 layer 开始
│
├─当前 token 的 MLA KV 按 resident slot_mapping 写 HBM
├─当前 token 的 Indexer K 按 original slot_mapping 写 HBM
│
├─LIDU
│   读完整 Indexer K
│   选 p，查/改 cache_slots，输出 p/s/miss_count/tail_info
│
├─KSC
│   只把 miss prefix 从 DRAM 写入 LIDU 指定的 resident slots
│
├─SFA-Offload
│   用 resident slots + HBM table 读 hit 和已换入 miss
│   再用 tail_info 读 dense tail
│   输出 [B,N1,512]
│
└─attention_finished
    如果当前 step 产生新满 block，整 block HBM → DRAM
```

当前实现是 `LIDU → KSC → SFA-Offload` 顺序执行。SFA-Offload 不会在 KSC 尚未完成时先单独计算 hit，也没有一个临时 miss buffer 让“hit 计算、miss 传输”在当前这条算子链中通算重叠。

直接调用顺序依据：`vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:480-558`，`vllm-glm/vllm_ascend/attention/sfa_v1.py:1119-1153,1226-1245`。

## 10. 源码阅读顺序

如果要对照代码再走一遍，建议按下列顺序：

1. `vllm-glm/vllm_ascend/attention/sfa_v1.py:1317-1555`：看当前 token 的三张 cache 写入、LIDU 调用与 SFA 交接。
2. `vllm-glm/vllm_ascend/dsa_sparse/dsa_input_batch_state.py:235-501`：看 `request_id/block_ids/stage` 如何投影为本轮请求行语义。
3. `vllm-glm/vllm_ascend/dsa_sparse/dsa_row_mode_runtime.py:539-905`：看哪些数据被 H2D，以及 HBM/DRAM table 如何 staging。
4. `vllm-glm/vllm_ascend/dsa_sparse/dsa_resident_pool.py:51-193`：看每层 `cache_slots` 的分配、行所有权与清理。
5. `vllm-glm/csrc/lightning_indexer_decode_update/lightning_indexer_decode_update_torch_adpt.h:12-134`：看 LIDU 完整 ABI。
6. `vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_vector.h:528-685,750-848,951-973`：看 hit/miss、淘汰和 first-fill。
7. `vllm-glm/csrc/kvcache_scatter_copy/op_kernel/kvcache_scatter_copy_kernel.h:52-140`：看 `(p,s)` 的两次 page translation 和单 token DMA。
8. `vllm-glm/csrc/sparse_flash_attention_for_offload/sparse_flash_attention_for_offload_torch_adpt.h:19-107`：看 SFA-Offload ABI。
9. `vllm-glm/csrc/sparse_flash_attention_for_offload/op_kernel/sparse_flash_attention_for_offload_service_cube_mla.h:61-100,523-538`：看 SFA 内部 resident slot → physical HBM 地址。
10. `vllm-glm/csrc/kv_cache_full_block_dump/op_kernel/kv_cache_full_block_dump.cpp:54-156`：看整 block HBM → DRAM 的多 core chunk copy。
