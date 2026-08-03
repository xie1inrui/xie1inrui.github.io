# Decode Top-K 选中后：KV 物理寻址、回载与 SFA 读取

> 本文只回答一件事：DSA 已经选出 Top-K 以后，一个 `token_id` 如何经过 block table 变成 KV cache 中的实际地址；KV 是否需要从 DRAM 回到 HBM；最后 SFA 用什么 index、什么 block table 读到这份 KV。
>
> 对比对象：原生 DSA-off 基线、旧 `v0.19.1rc1-gs`、当前 `vllm-ascend-v0.19.1rc1-gs-glm`。本文不展开 scheduler、block hash、H2D metadata 构造、远端 KV 或请求生命周期。

## 0. 先给结论：三个分支的输出并不完全相同

### 0.1 Top-K 输出的到底是什么

Top-K 输出的核心数据是最多 2048 个 INT32 整数索引；每个整数表示“请求历史序列中的一个 original token position”。它是被选中 Key/KV 的请求内位置标识。

例如：

```text
topk_indices[b,0,:] = [257, 1024, 89, ...]
```

含义是请求行 `b` 选中了历史序列位置 257、1024、89……对应的 Key/KV。算子此时没有输出这些位置上的 `[512]` NOPE_K 或 `[64]` ROPE_K 数据；后续 GS、KSC 或 SFA 才拿这些整数做地址翻译。

这个整数也不是全局唯一 key：

```text
请求内被选中项的标识 = (request/batch row, original token position)

若要唯一标识一份具体 KV payload，还至少需要：
(request/cache namespace, layer_id, original token position, KV plane)
```

因此同一个整数 `257` 在两个请求、两个 layer 中都可能出现，但指向的 KV payload 不同。有效历史不足 2048 时，剩余位置可以用 `-1` 表示无效/padding 项。

三个分支可以用同一个例子概括：

```text
原生：
topk_indices = 257
SFA 仍收到 257

旧 GS：
LI topk index = 257
GS 查 hit/miss 后把它放在 resident slot 33
SFA 收到 33，status[33] = 257 保留反向关系

当前 GLM：
topk_index = 257      # original token position，供 KSC 找 DRAM source
topk_slots = 33       # resident logical slot，供 KSC 写 HBM、供 SFA 读取
```

当前 GLM 的 `topk_index/topk_slots` tensor 容量是 16384，但不能把它理解为“输出了 16384 个 attention Key”。稳态稀疏 attention 的计算 Top-K 仍为 2048；更大的容量同时服务于 miss-pair 输出和 ENTER/first-fill resident 填充。


首先要避免把代码里的三种“key/index”混为一谈：

1. `token_id` / original token position：Top-K 选中的请求内历史位置，例如 `257`。
2. resident logical slot：卸载分支把选中 token 放入 resident HBM 后的位置，例如 `33`。
3. physical block ID：block table 查出的 cache arena 物理 block 编号，例如 `7` 或 `91`。它不是 token ID，也不是裸指针。

三个分支在“选中了哪个历史 token”这一层具有相同语义：输出值都是请求内 original token position；但 ABI、后处理结果以及最终交给 SFA 的 index 不相同。当前 LIDU 与原生 LI 也是不同算子，因此只能说输出语义一致，不能仅凭代码保证每次排序结果逐元素完全一致。

| 分支 | Top-K 算子刚完成时 | 稀疏行最终交给 SFA | 是否先回载 KV |
|---|---|---|---|
| 原生 DSA-off | `topk_indices`：INT32 `[B,1,2048]`，值为 original token position | 原样的 original token position | 否；完整 MLA KV 已在 HBM |
| 旧 `-gs` | 原生 LI 输出 original position；进入 GS 时规范为 INT32 `[B,1,1,2048]` | GS 输出 resident logical slot：先为 `[B,W]`，SFA 前变为 `[B,1,W]` | hit 不一定搬；miss 从 DRAM 搬到 resident HBM |
| 当前 `-gs-glm` | LIDU 同时输出 `topk_index`、`topk_slots`：INT32 `[B,1,16384]`，另有 `miss_count:[B]`、`tail_info:[B,2]` | SFA-Offload 接收 `topk_slots:[B,1,16384]`；稀疏计算只使用前 2048 个 slot，tail 由 `tail_info` 表示 | KSC 只搬 `[0, miss_count)` miss prefix |

这里的 `16384` 是 LIDU/KSC 的输出与首次填充容量，不代表 SFA 的 Top-K 从 2048 变成了 16384。当前 SFA-Offload 的固定 sparse compute count 仍是 2048。

还有一个非常关键的实现边界：

- block table 的值是 physical block ID。
- kernel 中算出的 `xxxAddr` 是相对 cache tensor 起点的扁平 element offset。
- 真正的 GM 字节地址是 `tensor_base_pointer + element_offset * element_size`。
- BF16/FP16 的 `element_size=2` 字节。

## 1. 三个分支共用的地址模型

本文统一使用以下符号：

| 符号 | 含义 | 当前实现 |
|---|---|---|
| `B` | Decode batch/request row 数 | 动态 |
| `K` | SFA 稀疏 Top-K 数 | `2048` |
| `C` | 当前 LIDU/KSC 输出容量 | `16384` |
| `BS` | MLA KV page/block 中的 token 数 | `128` |
| `D_nope` | latent NOPE_K 每 token 元素数 | `512` |
| `D_rope` | ROPE_K 每 token 元素数 | `64` |
| `p` | original token position | 示例取 `257` |
| `s` | resident logical slot | 示例取 `33` |

对于任意逻辑 token/slot `x`：

```text
logical_block(x) = x // 128
offset_in_block(x) = x % 128

physical_block = block_table[batch_row, logical_block(x)]  #batch_row表示的是 当前这个请求在这一轮decode中的行号，
block_table = [
  [12, 18, 25, ...],   # Request A
  [31, 37, 42, ...],   # Request B
  [ 5,  9, 16, ...],   # Request C
] 每一行代表该请求使用了哪些物理block，Req B的257token，位于batch_row=1 上的逻辑block2，则对应物理block42

NOPE element offset = (physical_block * 128 + offset_in_block(x)) * 512
ROPE element offset = (physical_block * 128 + offset_in_block(x)) * 64

byte address = tensor_base_pointer + element_offset * 2
               （BF16/FP16 都是 2 字节）
```

cache 在 vLLM/SFA 边界通常是：

```text
NOPE_K / latent KV : [num_blocks, 128, 1, 512]
ROPE_K             : [num_blocks, 128, 1,  64]
```

GS/KSC 的 copy adapter 会把 singleton KV-head 维 squeeze 掉，kernel 看到的是：

```text
NOPE_K : [num_blocks, 128, 512]
ROPE_K : [num_blocks, 128,  64]
```

两种 shape 的扁平地址公式相同，因为 KV head 数为 1。

### 1.1 page 粒度与 copy 粒度不是一回事

| 动作 | 寻址/管理粒度 | 实际 payload 粒度 |
|---|---|---|
| block table 映射 | 128-token page/block | 查出一个 physical block ID |
| 原生 SFA / SFA-Offload sparse index | token，`sparse_block_size=1` | SFA 按计算 tile 从 HBM GM 读入 L1/片上缓冲 |
| GS/KSC miss 回载 | 先按 128-token block 查地址 | 当前每个 Top-K miss 复制 1 个 token row：512+64 个元素 |
| HBM→DRAM 卸载 | physical block ID | 一次保存完整 128-token NOPE/ROPE block |

因此答案不是简单的“按 block”或“按 token”：**地址翻译通过 128-token page/block；Top-K 回载是 token 粒度；卸载是完整 block 粒度。**

### 1.2 SFA 前没有额外的 `[B,K,512]` HBM gather tensor

- 原生分支：完整 KV 本来就在 HBM，SFA 根据 `sparse_indices + block_table` 直接从 paged cache 读。
- 两个卸载分支：miss copy 先把 token row 原址写进 resident HBM cache；SFA 再根据 resident slot 读该 cache。
- SFA kernel 会把需要的 KV tile 从 HBM GM 搬到 L1/片上缓冲参与矩阵计算，但 Python 层没有再构造一份独立的、连续的 Top-K KV tensor。

---

## 2. 原生 DSA-off：original token position 直接定位完整 HBM KV

### 2.1 Decode 主流程

```text
Lightning Indexer 选出 original token p
→ LI 直接把 p 作为 sparse_indices 交给原生 SFA
→ SFA 内部拆分：
     logical_block = p // 128
     block_offset  = p % 128
→ 查 native MLA block_table：
     logical_block → HBM physical block
→ 从该 HBM physical block 的 block_offset
   读取单 token 的 NOPE_K / ROPE_K
→ SFA 在片上组织这些离散 KV，完成 attention
```

KV 存储线：

```text
完整 MLA KV 保留在 HBM paged cache
→ SFA 直接按 original token p 查表读取
```

### 2.2 SFA 接口上的数据

本表只列出本文寻址链涉及的关键数据，不是原生 SFA 的完整 ABI。

| 数据 | dtype | Decode shape | 算子接口角色 | 值的含义 | 一句话工作（沿用下文示例） |
|---|---|---|---|---|---|
| `sparse_indices` | INT32 | `[B,1,2048]` | LI 输出 → SFA 输入 | original token position `p` | 告诉 SFA 本轮选中了哪个原始位置，例如一项为 `p=257` |
| MLA `block_table` | INT32 | `[B,M_mla]` | SFA 输入 | original logical block → HBM physical block | 将 `257//128=2` 转为 HBM block，例如 `block_table[b,2]=37` |
| `key=value=kv_cache[0]` | BF16/FP16 | `[N_hbm,128,1,512]` | SFA 的 key/value 输入 | latent NOPE_K；key/value alias 同一 tensor | 保存并提供真实 latent KV，例如读取 HBM block `37` 的 offset `1`、共 512 个元素 |
| `key_rope=kv_cache[1]` | BF16/FP16 | `[N_hbm,128,1,64]` | SFA 的 key_rope 输入 | ROPE_K | 提供同一个 token 的 RoPE K，例如读取 HBM block `37` 的 offset `1`、共 64 个元素 |
| `query=ql_nope` | BF16/FP16 | `[B,H_local,512]` | SFA 的 query 输入 | 当前 Decode query | 与选中 token 的 latent KV 做 QK/加权计算，得到当前 Decode attention |
| `query_rope=q_pe` | BF16/FP16 | `[B,H_local,64]` | SFA 的 query_rope 输入 | query RoPE 部分 | 与对应的 64 维 ROPE_K 一起形成位置相关的注意力分数 |
| `attn_output` | BF16/FP16 | `[B,H_local,512]` | SFA 输出 | SFA 输出，与 query shape/dtype 一致 | 保存 SFA 对全部选中 KV 完成 softmax 和 V 聚合后的结果 |

原生 SFA kernel 先做：

```text
blockBegin = sparse_indices * sparse_block_size
           = p * 1
```

随后 `DataCopyPA` 做真正的 page translation：

```text
block_col = p / 128
block_off = p % 128
phys      = block_table[b, block_col]
```

### 2.3 具体例子：`token_id=257`

以下 block-table 数值是为了说明公式而设定的示例，不是运行日志：

```text
p = 257
native_mla_block_table[b, 2] = 37  #当前选中的indexer 对应的mla KV cache放在物理block 37上
```

第一步，拆出 logical block 与 block 内偏移：

```text
logical_block = 257 // 128 = 2
block_offset  = 257 % 128  = 1
```

第二步，查 native MLA block table：

```text
physical_hbm_block = native_mla_block_table[b, 2] = 37  
physical_token_row = 37 * 128 + 1 = 4737         # 在 cache tensor 中的扁平 token 行号
```

第三步，计算两张 cache plane 的偏移：

```text
NOPE element offset = 4737 * 512 = 2,425,344
ROPE element offset = 4737 *  64 =   303,168

BF16/FP16 NOPE byte offset = 2,425,344 * 2 = 4,850,688 bytes
BF16/FP16 ROPE byte offset =   303,168 * 2 =   606,336 bytes
```

所以 SFA 实际物理地址读取：

```text
NOPE_K base + 4,850,688 bytes，连续取 512 个 BF16/FP16 元素
ROPE_K base +   606,336 bytes，连续取  64 个 BF16/FP16 元素
```

`topk_indices` 中的值始终是 `257`；它不会被 LI 转成 physical block，也不会被转成 resident slot。physical address translation 完全发生在 SFA kernel 内。

### 2.4 代码证据

- LI 输出直接进入原生 SFA：`vllm-glm/vllm_ascend/attention/sfa_v1.py:1190-1264`。
- 原生 SFA 把 `sparse_indices` 乘 `sparse_block_size=1`：`vllm-glm/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_kernel_mla.h:900-918`。
- `DataCopyPA` 的 `token -> block_table -> physical block -> offset`：`vllm-glm/csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_service_cube_mla.h:61-100`。
- 输出按 `query.sizes()` 创建：`vllm-glm/csrc/sparse_flash_attention/sparse_flash_attention_torch_adpt.h:20-61`。

### 2.5 SFA 内如何组织离散 KV

原生路径是**按 token 粒度选择、在 SFA 内部查 MLA table**：LI 只交付 original position；SFA kernel 对这些离散 index 逐项做 page translation，将对应的单 token NOPE/ROPE row（实现中还会按 D 维切片）直接从 HBM GM 搬入 L1/片上连续计算 tile，再统一完成 QK、softmax 和 V。这里没有 Python/Host `BatchGet`，也不先生成一份独立的 `[B,2048,512]` 连续 HBM tensor；一次 SFA kernel 调用在内部完成 gather-on-read 和计算。

源码落点：原生 SFA 组织 L1 tile 见 `sparse_flash_attention_service_cube_mla.h:645-715`。

---

## 3.  `v0.19.1rc1-gs`：original token → GS hit/miss → resident slot → 原生 SFA

### 3.1 Decode 主流程、Offload 与取回

Decode 主流程：

```text
Lightning Indexer 选出 original token p
→ GS 查询 resident status，判断 p 是否已在 resident HBM
  │
  ├─ 原位 hit
  │    → 保留原 resident slot s，不搬 KV
  │
  ├─ hit，但当前有效 slots 存在空洞
  │    → HBM old slot → HBM new slot
  │    → 将有效 KV 压实到 resident slot 前缀
  │
  └─ miss
       → 为 p 选择 resident slot s
       → 从 DRAM 取回单 token KV，写入 slot s

→ GS 更新 status[s] = p
→ attention_indices_out 写入 resident slot s
→ 原生 SFA 接收 s，而不是 original token p
→ SFA 用 selection_block_table 将 s 翻译为 HBM 物理地址
→ 读取 resident NOPE_K / ROPE_K，完成 attention
```

Offload 线（HBM→DRAM）：
```text
一个 128-token MLA block 填满
→ 当前层 SFA 完成
→ DSAHotKVStore 根据 HBM block ID 取出完整 NOPE_K / ROPE_K
→ 分配或复用 DRAM physical block
→ 完整 block copy_ 到 NPU 可寻址的 swapped DRAM arena
→ full_block_table 记录 original logical block → DRAM physical block
```

取回线（DRAM→resident HBM）：
```text
GS 判断 original token p 为 miss
→ p 查 full_block_table，得到 DRAM source block/offset
→ resident slot s 查 selection_block_table，得到 HBM destination block/offset
→ 只复制 p 对应的一个 token：NOPE_K 512 + ROPE_K 64 个元素
→ status[s] = p
→ SFA 使用 s 读取刚写入的 resident KV
```

因此本分支是**按完整 block 卸载、按单 token 命中和回载**。BF16/FP16 下，一个完整 MLA block 的 NOPE/ROPE 合计为 `144 KiB / layer`；单 token 的精确回载量见 3.3。

### 3.2 GS 用到的数据与 shape

本表只列出本文 hit/miss、取回和 SFA handoff 涉及的关键数据，不是扩展 GS 的完整 ABI。

| 数据 | dtype | GS kernel shape | 算子接口角色 | 含义 | 一句话工作（沿用下文示例） |
|---|---|---|---|---|---|
| `selection_topk_indices` | INT32 | `[B,1,1,2048]` | LI 输出 → GS 输入 | LI 选出的 original token positions | 给出本轮要找的原始 token，例如 Top-K 项的值是 `p=257` |
| `resident_slot_token_status` | INT32 | 每层 view 约为 `[pool,1,1,K+1]` | GS 输入/输出（原地更新） | `status[slot]=original token position`；最后一项保存有效长度 | 将 resident slot 转回其中保存的原始 token，例如 `status[33]=257`，据此判断 hit/miss |
| `full_block_table` | INT32 | `[B,M_dram]` | GS 输入 | original logical block → DRAM physical block | 将原始逻辑块转换为 DRAM 物理块，例如 `full_block_table[b,2]=91` |
| `selection_block_table` | INT32 | `[B,M_hbm]` | GS 输入；随后也是 SFA 输入 | resident logical block → HBM physical block | 将 resident 逻辑块转换为 HBM 物理块，例如 `selection_block_table[b,0]=7` |
| DRAM NOPE/ROPE | BF16/FP16 | `[N_dram,128,512/64]` | GS 的 full KV 输入/source | 完整历史的 swapped-memory arena | 保存卸载后的真实 KV，例如 token `257` 位于 DRAM block `91` 的 offset `1` |
| resident HBM NOPE/ROPE | BF16/FP16 | `[N_hbm,128,512/64]` | GS 输入/输出（原地写入）；随后作为 SFA 输入 | SFA 实际读取的 resident cache | 保存回载后供 SFA 读取的 KV，例如 token `257` 被放到 HBM block `7` 的 offset `33` |
| `attention_indices_out` | INT32 | `[B,W]` | GS 输出 → SFA 输入 | sparse row 为 resident slots；SFA 前变为 `[B,1,W]` | 将本轮 Top-K 改写为 SFA 可读的 resident slot，例如把原始 token `257` 改写成 slot `33` |

### 3.3 miss 的精确 source/destination 地址

当前 GS tiling 中：

```text
selTopKBlockSize = 1
fullKvBlockSize  = 128
selKvBlockSize   = 128
```

因此 original token `p` 的 DRAM source 为：

```text
src_logical_block = p // 128
src_block_offset  = p % 128
src_phys_block    = full_block_table[b, src_logical_block]

src_nope = (src_phys_block * 128 + src_block_offset) * 512
src_rope = (src_phys_block * 128 + src_block_offset) * 64
```

GS 选择的 resident slot `s` 的 HBM destination 为：

```text
dst_logical_block = s // 128
dst_block_offset  = s % 128
dst_phys_block    = selection_block_table[b, dst_logical_block]

dst_nope = (dst_phys_block * 128 + dst_block_offset) * 512
dst_rope = (dst_phys_block * 128 + dst_block_offset) * 64
```

miss copy 的长度是：

```text
NOPE : 1 * 512 elements = 1,024 bytes（BF16/FP16）
ROPE : 1 *  64 elements =   128 bytes（BF16/FP16）
合计 : 1,152 bytes / miss token / layer
```

复制完成后，GS 更新 slot 身份并把 resident slot 交给 SFA。例如 `p=257`、`s=33`、`full_block_table[b,2]=91`、`selection_block_table[b,0]=7`：

```text
status[33] = 257
attention_indices_out 中写入 resident slot 33，而不是 original token 257

SFA 再做：
33 // 128 = 0
33 % 128  = 33
selection_block_table[b,0] = 7

因此 SFA 读到 GS 刚写入的 HBM physical block 7、token offset 33。
```

### 3.4 hit 时例子如何变化

若进入本轮前已经有：

```text
status[33] = 257
```

则 GS 找到 token `257` 的 resident slot `33`：

- 原位 hit：kernel 的 `CUR_SEG_HIT_FLAG` 分支直接 `continue`，不读 DRAM，也不复制 HBM。
- 需要压实时：用旧 resident slot 经 `selection_block_table` 算 HBM source，再复制到新 slot 对应的 HBM destination。
- 无论哪种 hit，SFA 最终仍使用 resident slot，而不是 original token position。

稀疏 row 的 `WriteSparseAttention()` 通常输出 `0..budget-1` 及 resident tail slots；所以 slot `33` 会作为 resident index 集合中的一项被 SFA 读取。DENSE row 是例外：GS 保留 original positions，并配合该 row 的完整 HBM block table，不做 DRAM 回载。

### 3.5 代码证据

- LI → GS → SFA index handoff：`vLLM-ascend-DSA-vllm-ascend-v0.19.1rc1-gs/vllm_ascend/attention/sfa_v1.py:1425-1510,1561-1590`。
- GS adapter 的 rank4 Top-K、rank3 cache 与输出 `[B,W]`：`vLLM-ascend-DSA-vllm-ascend-v0.19.1rc1-gs/vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py:158-189,221-400`。
- resident slot 输出：`vLLM-ascend-DSA-vllm-ascend-v0.19.1rc1-gs/csrc/gather_selection_kv_cache/op_kernel/gather_selection_kv_cache_split_bs_reuse_vec.h:260-318`。
- in-place hit、miss/compaction 分支、destination 地址：同一 kernel `:610-669`。
- DRAM source 地址与 token copy：同一 kernel `:674-712`。
- HBM compaction source 地址：同一 kernel `:714-754`。
- full-block dump 在 attention 后发生：`vLLM-ascend-DSA-vllm-ascend-v0.19.1rc1-gs/vllm_ascend/dsa_sparse/dsa_sparse.py:738-792`。
- 旧 store 的完整 block HBM→DRAM copy：`vLLM-ascend-DSA-vllm-ascend-v0.19.1rc1-gs/vllm_ascend/dsa_sparse/dsa_hot_kv_store_core.py:620-790`。

---

## 4. 当前 `-gs-glm`：LIDU miss indication → KSC token copy → SFA-Offload

### 4.1 Decode 主流程、Offload 与取回

Decode 主流程：

```text
LIDU 在完整 Indexer K 上选出 original token p
→ 查询 cache_slots[p]
  │
  ├─ cache_slots[p] = s
  │    → resident HBM hit，保留现有 slot s，不搬 KV
  │
  └─ cache_slots[p] = -1
       → resident HBM miss
       → 淘汰一个不属于本轮 Top-K 的 resident token
       → 复用其 resident slot s，并更新 original token → slot 映射
       → 生成 miss pair (p, s)

→ LIDU 输出：
     topk_index = original token p
     topk_slots = resident slot s
     miss_count = miss pair 数量
     tail_info  = dense tail 的起点和长度

→ KSC 只处理 [0, miss_count) 的 miss pairs
→ hit 不经过 KSC 搬运
→ miss KV 已写入该写入哪个resident slot 已被LIDU 计算好

→ SFA-Offload 接收 topk_slots
→ 前 2048 个 slot 用于 sparse attention
→ 每个 slot s 查 HBM block table：
     s → resident logical block/offset → HBM physical address
→ 读取 resident HBM NOPE_K / ROPE_K
→ 再根据 tail_info 连续读取 dense tail
→ 完成 attention
```

Offload 线（HBM→DRAM）：
```text
一个 128-token MLA block 在当前 Decode step 填满
→ 当前层 SFA-Offload 完成
→ KvCacheFullBlockDump 接收 HBM source / DRAM destination physical block IDs
→ 完整复制 NOPE_K / ROPE_K 到 swapped DRAM
→ 记录 original logical block → DRAM physical block
→ 下一次 Decode 才进入可选择、可取回的历史 candidate
```

取回线（DRAM→resident HBM）：
```text
LIDU 输出 miss pair (p, s)
→ KSC 用 p 查 dram_block_table，得到 DRAM source block/offset
→ KSC 用 s 查 hbm_block_table，得到 HBM destination block/offset
→ 只复制该 miss token 的 NOPE_K / ROPE_K 到 resident slot s
→ KSC 原址更新 resident HBM，不创建新的 KV tensor
→ SFA-Offload 使用同一个 s 和 HBM block table 读取 KV
```

Dense tail 旁路线：
```text
未满 128-token 的 tail 保留在 HBM
→ 不参与 LIDU resident hit/miss，也不经过 KSC
→ SFA-Offload 根据 tail_info 连续读取
```

Full-block dump 每次接收的已经是 physical source/destination block IDs，kernel 内不再查 request block table；`dst_block_id=-1` 是 graph no-op sentinel，block 0 仍是有效地址。BF16/FP16 下，完整 block 的 NOPE/ROPE 合计为 `144 KiB / layer`。

### 4.2 LIDU/KSC/SFA-Offload 的 shape 合约

本表只列出本文 LIDU→KSC→SFA-Offload 数据链的关键参数，不是三个算子的完整 ABI。

| 数据 | dtype | shape | 算子接口角色 | 消费者与含义 | 一句话工作（沿用下文示例） |
|---|---|---|---|---|---|
| `topk_index` | INT32 | `[B,1,16384]` | LIDU 输出 → KSC 输入 | KSC source token IDs；值为 original positions | 告诉 KSC 要从 DRAM 取哪个原始 token，例如 miss 项为 `p=257` |
| `topk_slots` | INT32 | `[B,1,16384]` | LIDU 输出 → KSC 输入；随后作为 SFA-Offload 输入 | KSC destination slots；随后也是 SFA-Offload sparse indices | 指定 KSC 写入和 SFA 读取的 resident slot，例如 `s=33` |
| `miss_count` | INT32 | `[B]` | LIDU 输出 → KSC 输入 | 每行有效 miss pair 前缀长度 | 限定需要真正搬运的前缀；例如值为 `5` 时 KSC 只处理前 5 对 `(p,s)` |
| `tail_info` | INT32 | `[B,2]` | LIDU 输出 → SFA-Offload 输入 | `[tail_slot_start, tail_token_count]` | 告诉 SFA-Offload 从哪个 resident slot 起连续读取多少个 dense-tail token |
| `dram_block_table` | INT32 | `[B,M_dram]` | KSC 输入 | original logical block → DRAM physical block | 将 `p=257` 的 logical block `2` 转为 DRAM block，例如 `dram_block_table[b,2]=91` |
| `hbm_block_table` | INT32 | `[B,M_hbm]` | KSC 输入；对应 HBM table view 也是 SFA-Offload 输入 | resident logical block → HBM physical block | 将 `s=33` 的 resident block `0` 转为 HBM block，例如 `hbm_block_table[b,0]=7` |
| DRAM NOPE/ROPE | BF16/FP16 | `[N_dram,128,512/64]` | KSC 输入/source | KSC source | 保存完整历史 KV；KSC 从 DRAM block `91` 的 offset `1` 读取 token `257` |
| resident HBM NOPE/ROPE | BF16/FP16 | `[N_hbm,128,512/64]` | KSC 输入/输出（原地写入）；随后作为 SFA-Offload 输入 | KSC 原址 destination；SFA 侧为 `[N_hbm,128,1,512/64]` view | KSC 把 token `257` 写到 HBM block `7` 的 offset `33`，随后 SFA 从同一位置读取 |
| SFA query/query_rope | BF16/FP16 | `[B,H_local,512/64]` | SFA-Offload 输入 | 当前 Decode query | 与 `topk_slots[:2048]` 和 dense tail 指向的 resident KV 进行注意力计算 |
| SFA output | BF16/FP16 | `[B,H_local,512]` | SFA-Offload 输出 | 与 query shape/dtype 相同 | 保存 SFA-Offload 汇聚 sparse Top-K 与 dense tail 后的 attention 结果 |

`topk_index/topk_slots` 的 16384 容量有两个用途：steady sparse 保存 Top-K 与 miss pairs；ENTER/first-fill 可以一次构造更多 resident copy pairs。SFA-Offload 并不会对 16384 项全部做稀疏 attention：有合法 tail 时固定读取 2048 个 sparse slots，再读取 `tail_info` 指定的连续 tail。

### 4.3 KSC 的精确地址公式

这里有两张名字容易混淆的 HBM table view：

- KSC 显式接收 `forward_batch.batch_hbm_block_table`；它从 vLLM 当前 MLA attention group 的 block-table row staging 而来。
- SFA-Offload 接收 `attn_metadata.full_block_tables`（存在时）或 `attn_metadata.block_table`。

二者在该 row 的执行合约中都描述“resident/original logical slot → 当前 MLA HBM physical block”；它们可能是不同的 tensor view，不能用 Python object identity 判断，但 SFA 必须能够用 KSC 写入时的同一逻辑 slot 找回相同 physical HBM block。名称中的 `full_block_tables` 也不要误认为 DRAM table：**DRAM source table 只作为 `batch_dram_block_table` 传给 KSC，不会传给 SFA。**

KSC kernel 固定：

```text
BLOCK_SIZE  = 128
KV_CACHE_DIM = 512
K_ROPE_DIM   = 64
```

对于 miss prefix 中的一个 pair：

```text
p = topk_index[b,0,i]
s = topk_slots[b,0,i]
要求 i < miss_count[b]
```

source 与 destination：

```text
src_block_col    = p >> 7 = p // 128
src_block_offset = p & 127 = p % 128
src_phys_block   = dram_block_table[b, src_block_col]

dst_block_col    = s >> 7 = s // 128
dst_block_offset = s & 127 = s % 128
dst_phys_block   = hbm_block_table[b, dst_block_col]

src_nope = (src_phys_block * 128 + src_block_offset) * 512
src_rope = (src_phys_block * 128 + src_block_offset) * 64
dst_nope = (dst_phys_block * 128 + dst_block_offset) * 512
dst_rope = (dst_phys_block * 128 + dst_block_offset) * 64
```

KSC 依次把 DRAM NOPE/ROPE 读到本核 local buffer，再写入 resident HBM NOPE/ROPE。

### 4.4 具体例子：`token_id=257` 是 miss

使用与旧 GS 完全相同的示例表值：

```text
topk_index[b,0,i] = 257
topk_slots[b,0,i] = 33
i < miss_count[b]

dram_block_table[b,2] = 91
hbm_block_table[b,0]  = 7
```

KSC source：

```text
257 >> 7  = 2
257 & 127 = 1
dram_block_table[b,2] = 91

src physical row = 91 * 128 + 1 = 11,649
src NOPE element offset = 5,964,288
src ROPE element offset =   745,536
```

KSC destination：

```text
33 >> 7  = 0
33 & 127 = 33
hbm_block_table[b,0] = 7

dst physical row = 7 * 128 + 33 = 929
dst NOPE element offset = 475,648
dst ROPE element offset =  59,456
```

KSC 完成：

```text
DRAM token p=257
  NOPE 512 elements + ROPE 64 elements
        │
        ▼
resident HBM slot s=33
  physical block 7, offset 33
```

随后 Python 不再传 `topk_index=257` 给 SFA，而是：

```text
DSAOffloadSelectionOutput.sparse_indices = topk_slots
sparse_indices[b,0,j] = 33
```

SFA-Offload 对 `s=33` 再做一次 paged address translation：

```text
logical resident block = 33 // 128 = 0
resident block offset  = 33 % 128  = 33
SFA HBM block_table[b,0] = 7

SFA 读取：
NOPE element offset = (7 * 128 + 33) * 512 = 475,648
ROPE element offset = (7 * 128 + 33) *  64 =  59,456
```

这与 KSC 的 destination 完全相同，所以数据链闭合：

```text
topk_index 257
  -> DRAM block 91 / offset 1
  -> KSC copy
  -> HBM block 7 / offset 33
  -> topk_slots 33
  -> SFA-Offload block_table
  -> HBM block 7 / offset 33
```

### 4.5 DENSE 与 tail 特殊路径

#### DENSE row

LIDU 的 `WriteNonOffloadedOutput()` 把前 2048 项 original positions 同时写进 `topk_index` 与 `topk_slots`，并令 `miss_count=0`。因此 KSC no-op，SFA-Offload 把这些值解释为完整 HBM cache 中的 original positions。也就是说，当前分支的 DENSE row 在地址语义上退化为原生路径。

#### dense tail

SFA-Offload 不要求把 tail 每个 slot 重复写入 16384 项数组。它读取：

```text
tail_info[b] = [tail_slot_start, tail_token_count]
real_tail_slot = tail_slot_start + tail_offset
```

再用同一套 `slot // 128`、`slot % 128`、HBM block table 公式读取连续 resident tail。

### 4.6 代码证据

- LIDU → KSC → `DSAOffloadSelectionOutput`：`vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:480-565`。
- LIDU 四个输出的 adapter shape/dtype：`vllm-glm/csrc/lightning_indexer_decode_update/lightning_indexer_decode_update_torch_adpt.h:12-134`。
- `16384` output capacity 与 `2048` compute Top-K：`vllm-glm/vllm_ascend/dsa_sparse/dsa_types.py:20-29`、`vllm-glm/csrc/lightning_indexer_decode_update/op_kernel/lightning_indexer_decode_update_service_vector.h:29-31`。
- KSC cache/table/pair shape：`vllm-glm/csrc/kvcache_scatter_copy/kvcache_scatter_copy_torch_adpt.h:11-100`。
- KSC 的 `p/s -> block table -> element address` 与 token copy：`vllm-glm/csrc/kvcache_scatter_copy/op_kernel/kvcache_scatter_copy_kernel.h:91-140`。
- KSC HBM table 从 vLLM MLA block table staging：`vllm-glm/vllm_ascend/dsa_sparse/dsa_row_mode_runtime.py:640-679,838-856`。
- SFA-Offload Python handoff：`vllm-glm/vllm_ascend/attention/sfa_v1.py:1214-1264`。
- SFA-Offload ABI 与输出 shape：`vllm-glm/csrc/sparse_flash_attention_for_offload/sparse_flash_attention_for_offload_torch_adpt.h:18-105`。
- SFA-Offload 的 sparse 2048 + tail：`vllm-glm/csrc/sparse_flash_attention_for_offload/op_kernel/sparse_flash_attention_for_offload_kernel_mla.h:304-343`。
- SFA-Offload 的 resident slot page translation：`vllm-glm/csrc/sparse_flash_attention_for_offload/op_kernel/sparse_flash_attention_for_offload_service_cube_mla.h:61-100,511-540`。
- full-block dump 的时点与调用：`vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:590-645`。
- full-block dump 的 physical block copy：`vllm-glm/csrc/kv_cache_full_block_dump/op_kernel/kv_cache_full_block_dump.cpp:11-105`。

---

## 5. 用同一个 `token_id=257` 横向看三条链

下表中的 block-table 值均为本文构造的统一示例：

```text
native HBM table[b,2] = 37
DRAM table[b,2]       = 91
resident slot         = 33
resident HBM table[b,0] = 7
```

| 阶段 | 原生 DSA-off | 旧 GS | 当前 GLM |
|---|---|---|---|
| Top-K original position | `p=257` | `p=257` | `topk_index=257` |
| 第一次 logical block/offset | `2 / 1` | `2 / 1` | `2 / 1` |
| 第一次查表 | native HBM table → `37` | DRAM table → `91` | DRAM table → `91` |
| source physical row | `37*128+1=4737` | `91*128+1=11649` | `91*128+1=11649` |
| 回载动作 | 无 | GS miss copy 1 token | KSC miss-prefix copy 1 token |
| resident slot | 无 | `s=33` | `topk_slots=33` |
| resident logical block/offset | 无 | `0 / 33` | `0 / 33` |
| resident HBM physical block | 无 | HBM table → `7` | HBM table → `7` |
| resident physical row | 无 | `7*128+33=929` | `7*128+33=929` |
| SFA 收到的 index | `257` | `33` | `33` |
| SFA 最终读取 | HBM block 37 / offset 1 | HBM block 7 / offset 33 | HBM block 7 / offset 33 |

最短总结：

```text
原生：
257 -> native HBM block_table -> HBM KV -> SFA

旧 GS：
257 -> status hit/miss
    -> miss: DRAM block_table -> DRAM KV -> resident HBM slot 33
    -> GS 输出 33 -> resident HBM block_table -> SFA

当前 GLM：
topk_index 257 + topk_slots 33 + miss_count
    -> KSC: DRAM block_table -> DRAM KV -> resident HBM slot 33
    -> SFA-Offload 使用 topk_slots 33 -> HBM block_table -> SFA
```

## 6. 最终需要记住的四个边界

1. Top-K 返回的不是 KV tensor 地址，而是请求内 token position；physical block 必须在 GS/KSC/SFA 中通过 block table 查出。
2. 原生 SFA 收到 original position；两个卸载分支的稀疏 SFA 收到 resident slot。
3. 卸载保存完整 128-token block；Top-K miss 回载只复制被选中的单 token row。
4. GS/KSC 把 miss KV 写回 resident HBM 后，SFA 仍会用 resident slot 和 HBM block table 再做一次地址翻译；没有额外的 Python Top-K KV gather tensor。
