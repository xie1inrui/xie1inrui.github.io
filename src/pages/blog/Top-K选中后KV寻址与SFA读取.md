# Decode Top-K 选中后：KV 物理寻址、回载与 SFA 读取

> 本文主要回答：DSA 已经选出 Top-K 以后，一个 original token position 如何经过 block table 变成 KV cache 中的实际地址；KV 是否需要从 DRAM 回到 HBM；最后 SFA 用什么 index、什么 block table 读到这份 KV。文中示例旧称 `token_id=257`，实际均指请求内位置 `p=257`，不是词表 token ID。
>
> 对比对象：原生 DSA-off 基线、旧 `v0.19.1rc1-gs`、当前 `vllm-ascend-v0.19.1rc1-gs-glm`。为解释 shape 和 resident 状态，本文额外覆盖与本链路直接相关的请求行、MTP 限制、状态生命周期和关键路径；不展开远端 KV 或无关模型计算。

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

首先要避免把代码里的三种“key/index”混为一谈：

1. `token_id` / original token position：Top-K 选中的请求内历史位置，例如 `257`。
2. resident logical slot：卸载分支把选中 token 放入 resident HBM 后的位置，例如 `33`。
3. physical block ID：block table 查出的 cache arena 物理 block 编号，例如 `7` 或 `91`。它不是 token ID，也不是裸指针。

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

### 1.1 真正管理 resident HBM 的数据结构

每个启用 DSA 的 attention layer 都有独立的 cache tensor；单层 shape 不含 layer 维。真实 MLA KV 是 vLLM 在 worker/model-runner 初始化时分配到 NPU HBM 的 paged tensor：

```text
当前 layer 的 HBM resident cache （KV cache）
├─ NOPE/latent KV : [N_hbm, 128, 1, 512]
└─ ROPE_K         : [N_hbm, 128, 1,  64]

完整 Indexer K（另一组 cache）
└─ Indexer K      : [N_indexer, 128, 1, 128]
```

同一个 Decode token 逐层执行时，每层的 Indexer query、Indexer K 和聚合权重不同，因此 Top-K 和 resident hit/miss 也按层独立；各层不会共用同一份真实 KV payload。

`N_hbm` 是当前层共享 physical block pool 的 block 数，不是 batch。请求维由 `block_table[b,:]` 表达：`b` 选择请求行，逻辑 block 再映射到该层的 physical block。不同 layer 即使得到相同 physical block ID，也会因为 tensor/base pointer 不同而访问不同 KV。

卸载只改变 HBM 保存的容量和职责，不改变单个 block 的布局：

| 场景 | 单层 KV shape | 保存范围 |
|---|---|---|
| 原生 HBM | `[N_hbm,128,1,512/64]` | 当前已分配的完整历史 |
| 卸载分支 resident HBM | `[N_hbm,128,1,512/64]` | resident working set |
| 卸载分支 DRAM arena | `[N_dram,128,1,512/64]` | 已卸载的完整历史 block |

卸载不会在 Decode 中 reshape 或缩小已有 HBM tensor；`N_hbm` 的实际容量由初始化配置决定，卸载分支只是增加 DRAM backing，并把 HBM 作为 resident working set 使用。所有 arena 都按 layer、NOPE/ROPE plane 分开。GS/KSC adapter 的 `squeeze(2)` 仅把 `[N,128,1,D]` 变成 kernel 接收的 `[N,128,D]` view；base pointer 和 payload 不变。

| 对象 | 谁拥有/维护 | 保存什么 | 谁实际使用 |
|---|---|---|---|
| HBM NOPE/ROPE tensor | vLLM KV cache 分配器/model runner | 真正的 resident KV 字节 | GS/KSC 原址写，SFA 读 |
| HBM block table | vLLM KV manager，并在本轮按 request row staging | logical slot/block → HBM physical block | GS/KSC/SFA |
| `DSALayerCacheRegistry` | 当前 GLM worker | 每层 HBM tensor 的稳定引用和地址一致性检查；不分配 block | dump/KSC hook 查层 |
| 旧 GS `resident_slot_token_status` | resident pool 持有，GS 原址更新 | resident slot → original position | 下一 step 的 GS hit/miss |
| 当前 GLM `cache_slots` | resident pool 持有，LIDU 原址更新 | original position → resident slot | 下一 step 的 LIDU hit/miss |

因此 resident 管理分三层：vLLM 管物理 tensor/block，block table 管请求逻辑块到物理块的映射，GS/LIDU 状态只管 token 与 resident logical slot 的关系。

源码依据：HBM shape 由 `vllm-glm/vllm_ascend/attention/sfa_v1.py:118-125` 和 `vllm-glm/vllm_ascend/worker/model_runner_v1.py:3991-4025` 构造；当前 GLM 的逐层 cache registry 见 `vllm-glm/vllm_ascend/dsa_sparse/dsa_layer_cache_zones.py:20-39,147-184`。

### 1.2 `base_addr` 到底是什么

`base_addr` 就是**当前算子参数中、当前 layer、当前 cache plane tensor 的起始 data pointer**：

```text
NOPE address = current_layer_nope_base
             + NOPE element offset * element_size

ROPE address = current_layer_rope_base
             + ROPE element offset * element_size
```

它不是某个请求的起始地址，也不是整个模型所有 layer 的统一 KV 起始地址。请求和 token 的区别由 `batch_row + block_table + block_offset` 转成相对这个 base 的 element offset。SFA-Offload 将传入的 `key/key_rope` 直接绑定为 `keyGm/kRopeGm`，再由 `DataCopyPA` 查表计算 offset；见 `sparse_flash_attention_for_offload_kernel_mla.h:440-472` 和 `sparse_flash_attention_for_offload_service_cube_mla.h:62-100`。

### 1.3 Top-K shape、两个 singleton 维和 MTP

原生 Lightning Indexer 的 shape 规则取决于 query layout：

```text
BSND query : [B,S,N_idx,128] -> topk [B,S,N2,K]
TND  query : [T,N_idx,128]   -> topk [T,N2,K]
T = 当前 batch 中所有请求的 query token 总数
```

其中 `N_idx` 是 32/64 个 Indexer query heads；`N2` 是 Indexer K 的 KV head 数，GLM/MLA 当前为 `1`；`K=2048`。TND 会把多个请求的 query token 压平到 `T=sum(query_len)`，请求边界由 `actual_seq_lengths_query:[B]` 保留。

| 分支 | 普通 single-token Decode | singleton 维含义 | 开启 MTP 后的当前实现 |
|---|---|---|---|
| 原生 LI + 原生 SFA | `[B,1,2048]`，一般 TND ABI 为 `[T,1,2048]` | `1=N2`，即单 Indexer KV head；不是 sequence 维 | 原生 TND ABI 可以用增大的 `T` 表达多 query token，sequence 不会变成这个 `1` |
| 旧 GS 集成路径 | LI `[B,1,2048]`，GS 前规范为 `[B,1,1,2048]` | GS 的 rank-4 语义是 `[B,S,H,K]`，当前 `S=1,H=1` | **集成路径只允许 single-query Decode**；通用 GS kernel 虽能描述 `S>1`，当前 Python/status/runtime 没有接通 MTP |
| 当前 GLM LIDU | `[B,1,16384]` | LIDU 每 request 输出一条选择列表；这个 `1` 不是 seq，也不是 32/64 个 query heads | **DSA sparse offload 明确拒绝 MTP/multi-token row**，不会自动把该维改成 MTP token 数 |

旧 GS 的通用算子接口还支持 TND `[T,H,K]` 或 BSND `[B,S,H,K]`，并限制 `S<=8`；但是本分支持久 status 固定为 `S=H=1`，且 `sfa_v1.py` 要求 `query_position_rows.shape[1]==1`。因此不能根据通用算子能力推导当前 vLLM 集成已经支持 MTP。当前 GLM 更严格：LIDU query ABI 本身是 `[B,N_idx,128]`，且 model runner 对 DSA sparse rows 要求 `attn_state=DecodeOnly`、每请求恰好一个 scheduled token。

源码依据：原生 LI shape 见 `lightning_indexer_vllm_torch_adpt.h:43-53`；GS 的 `[B,S,H,K]` 解析和 `S<=8` 见旧分支 `gather_selection_kv_cache_tiling.cpp:48-53,176-244`；旧集成 single-query gate 见旧分支 `sfa_v1.py:1290-1306,1470-1477`；当前 GLM 的 single-token gate 见 `model_runner_v1.py:838-852` 和 `dsa_sparse.py:403-412`。

### 1.4 page 粒度与 copy 粒度不是一回事

| 动作 | 寻址/管理粒度 | 实际 payload 粒度 |
|---|---|---|
| block table 映射 | 128-token page/block | 查出一个 physical block ID |
| 原生 SFA / SFA-Offload sparse index | token，`sparse_block_size=1` | SFA 按计算 tile 从 HBM GM 读入 L1/片上缓冲 |
| GS/KSC miss 回载 | 先按 128-token block 查地址 | 当前每个 Top-K miss 复制 1 个 token row：512+64 个元素 |
| HBM→DRAM 卸载 | physical block ID | 一次保存完整 128-token NOPE/ROPE block |

因此答案不是简单的“按 block”或“按 token”：**地址翻译通过 128-token page/block；Top-K 回载是 token 粒度；卸载是完整 block 粒度。**

### 1.5 SFA 前没有额外的 `[B,K,512]` HBM gather tensor

原生 SFA 在一次 kernel 调用内根据 `sparse_indices + block_table` 完成 page translation 和 HBM GM→L1 的 gather-on-read；两个卸载分支先把 miss 原址写入 resident HBM，再由 SFA 按 resident slot 做同样的读取。Python/Host 都不会构造独立的连续 `[B,K,512]` KV tensor，也不存在额外 `BatchGet`。

### 1.6 三个分支的依赖关键路径

这里的“关键路径”指代码和 stream 依赖，不代表已经通过 profiler 证明哪一个算子耗时最长：

```text
原生：LI → SFA

旧 GS：LI → GS（hit/miss、必要 copy、status 更新）→ SFA

当前 GLM：LIDU（选择并更新 cache_slots）
          → KSC（完成全部 miss DRAM→目标 HBM slot）
          → SFA-Offload（读取完整 sparse slots + tail）
```

满 block dump 在当前层 attention 后发射，不是当前 SFA 的输入依赖；当前 GLM 的同流时序及尚未实现的 hit/miss 重叠详见 4.4。

---

## 2. 原生 DSA-off：original token position 直接定位完整 HBM KV

### 2.1 Decode 主流程

```text
Lightning Indexer 选出 original token p
→ LI 直接把 p 作为 sparse_indices 交给原生 SFA
→ 完整 MLA KV 始终保留在 HBM paged cache
→ SFA 内部拆分：
     logical_block = p // 128
     block_offset  = p % 128
→ 查 native MLA block_table：
     logical_block → HBM physical block
→ 从该 HBM physical block 的 block_offset
   读取单 token 的 NOPE_K / ROPE_K
→ SFA 在片上组织这些离散 KV，完成 attention
```

### 2.2 SFA 接口上的数据

本表只列出本文寻址链涉及的关键数据，不是原生 SFA 的完整 ABI。

| 数据 | dtype | Decode shape | 算子接口角色 | 值的含义 | 一句话工作（沿用下文示例） |
|---|---|---|---|---|---|
| `sparse_indices` | INT32 | 普通 Decode `[B,1,2048]`；一般 TND `[T,1,2048]` | LI 输出 → SFA 输入 | original token position `p` | 告诉 SFA 每个 query token 选中了哪些原始位置，例如一项为 `p=257` |
| MLA `block_table` | INT32 | `[B,M_mla]` | SFA 输入 | original logical block → HBM physical block | 将 `257//128=2` 转为 HBM block，例如 `block_table[b,2]=37` |
| `key=value=kv_cache[0]` | BF16/FP16 | `[N_hbm,128,1,512]` | SFA 的 key/value 输入 | latent NOPE_K；key/value alias 同一 tensor | 保存并提供真实 latent KV，例如读取 HBM block `37` 的 offset `1`、共 512 个元素 |
| `key_rope=kv_cache[1]` | BF16/FP16 | `[N_hbm,128,1,64]` | SFA 的 key_rope 输入 | ROPE_K | 提供同一个 token 的 RoPE K，例如读取 HBM block `37` 的 offset `1`、共 64 个元素 |
| `query=ql_nope` | BF16/FP16 | 普通 Decode `[B,H_local,512]`；一般 TND `[T,H_local,512]` | SFA 的 query 输入 | 当前 Decode query | 与选中 token 的 latent KV 做 QK/加权计算，得到当前 Decode attention |
| `query_rope=q_pe` | BF16/FP16 | 普通 Decode `[B,H_local,64]`；一般 TND `[T,H_local,64]` | SFA 的 query_rope 输入 | query RoPE 部分 | 与对应的 64 维 ROPE_K 一起形成位置相关的注意力分数 |
| `attn_output` | BF16/FP16 | 与 query 相同，普通 Decode `[B,H_local,512]` | SFA 输出 | SFA 输出，与 query shape/dtype 一致 | 保存 SFA 对全部选中 KV 完成 softmax 和 V 聚合后的结果 |

这里 `[B,1,2048]` 中的 `1` 是 Indexer K 的单 KV head `N2=1`，不是 sequence length。普通 Decode 每请求一个 query token，所以 TND 的 `T=B`；MTP/多 token query 时，原生 ABI 把它们展平到更大的 `T`，请求边界由 `actual_seq_lengths_query:[B]` 描述。

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
- SFA 将离散 KV 组织为 L1 计算 tile：同一文件 `:645-715`。
- 输出按 `query.sizes()` 创建：`vllm-glm/csrc/sparse_flash_attention/sparse_flash_attention_torch_adpt.h:20-61`。

---

## 3. `v0.19.1rc1-gs`：original token → GS hit/miss → resident slot → 原生 SFA

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
| `selection_topk_indices` | INT32 | 当前集成 `[B,1,1,2048]`；通用 GS 为 BSND `[B,S,H,K]` 或 TND `[T,H,K]` | LI 输出 → GS 输入 | LI 选出的 original token positions | 给出每个 query token 本轮要找的原始 token，例如一项为 `p=257` |
| `resident_slot_token_status` | INT32 | 总存储 `[L,pool,1,1,K+1]`；每层 view `[pool,1,1,K+1]` | GS 输入/输出（原地更新） | `status[pool_row,0,0,s]=p`；最后一项保存有效 resident 长度 | 将 resident slot 转回其中保存的原始 token，例如 `status[r,0,0,33]=257`，据此判断 hit/miss |
| `full_block_table` | INT32 | `[B,M_dram]` | GS 输入 | original logical block → DRAM physical block | 将原始逻辑块转换为 DRAM 物理块，例如 `full_block_table[b,2]=91` |
| `selection_block_table` | INT32 | `[B,M_hbm]` | GS 输入；随后也是 SFA 输入 | resident logical block → HBM physical block | 将 resident 逻辑块转换为 HBM 物理块，例如 `selection_block_table[b,0]=7` |
| DRAM NOPE/ROPE | BF16/FP16 | `[N_dram,128,512/64]` | GS 的 full KV 输入/source | 完整历史的 swapped-memory arena | 保存卸载后的真实 KV，例如 token `257` 位于 DRAM block `91` 的 offset `1` |
| resident HBM NOPE/ROPE | BF16/FP16 | `[N_hbm,128,512/64]` | GS 输入/输出（原地写入）；随后作为 SFA 输入 | SFA 实际读取的 resident cache | 保存回载后供 SFA 读取的 KV，例如 token `257` 被放到 HBM block `7` 的 offset `33` |
| `attention_indices_out` | INT32 | `[B,W]` | GS 输出 → SFA 输入 | sparse row 为 resident slots；SFA 前变为 `[B,1,W]` | 将本轮 Top-K 改写为 SFA 可读的 resident slot，例如把原始 token `257` 改写成 slot `33` |

当前 `[B,1,1,2048]` 按 GS 通用 rank-4 ABI 应读作 `[B,S,H,K]`，即 `S=1` 个 query position、`H=1` 个 KV head。上游原生 LI 的 TND 输出本来是 `[B,H,K]=[B,1,2048]`，wrapper 再插入一个 singleton 变成 rank-4；因为当前 `S` 和 `H` 都是 `1`，两个轴即使在 wrapper 注释中次序写法不同，也不影响地址。MTP 时 `S>1` 后二者不能再互换，必须显式恢复 `[B,S,H,K]` 和对应的 status/block-table row；本分支集成代码没有完成这项工作，并以 single-query gate 拒绝该路径。

`resident_slot_token_status` 也不是 KV payload。它由 `DSAResidentTokenPool` 在初始化时一次分配，按 `[layer,pool_row,S=1,H=1,K+1]` 跨 decode step 保留；GS 根据 `req_pool_entries[b]=pool_row` 选择当前请求的状态行并原址修改，KSC/SFA 类算子不会自动发现这张表。请求结束或 preempt 时，Python pool 会把该请求所有 layer 的状态行清为 `-1`。源码见旧分支 `dsa_resident_pool.py:30-76,82-107,133-188` 和 `dsa_ascend_ops_backend.py:267-280,370-400`。

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

> 如需从当前 token 的 Indexer/MLA KV 生成、vLLM block ID 与 H2D metadata，一直跟到 LIDU/KSC/SFA-Offload 的全 shape 链路，见[当前 GLM 分支 KV 完整数据流](./当前GLM分支KV完整数据流.md)。

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
→ LIDU 只决定 miss 应写入哪个 resident slot
→ KSC 完成后，miss KV 才真正出现在该 resident HBM slot

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
| LIDU `query` | BF16/FP16 | `[B,N_idx,128]`，`N_idx=32/64` | LIDU 输入 | 每个 request row 的 Indexer query heads | 对该请求的完整 Indexer K 历史打分；`N_idx` 不会原样出现在 Top-K 输出 shape 中 |
| LIDU `key` | BF16/FP16 | `[N_indexer,128,1,128]` | LIDU 输入 | 完整 HBM Indexer K cache | 借助 Indexer block table 提供所有 candidate K；这里的 `1` 是单 KV head |
| LIDU `weights` | BF16/FP16 | `[B,N_idx]` | LIDU 输入 | Indexer head 聚合权重 | 将 32/64 个 head 的 score 聚合为每个 request 一条 Top-K 列表 |
| `req_pool_entries` | INT32 | `[B]` | LIDU 输入 | 当前 batch row → 持久 resident pool row | 例如 `req_pool_entries[b]=r`，让 LIDU 访问第 `r` 行 `cache_slots` |
| `cache_slots` | INT32 | 总存储 `[L,pool,W]`；LIDU 每层输入 `[pool,W]` | LIDU 输入/输出（原址更新） | 前 `W-1` 列为 original position `p` → resident slot `s`；末列为 budget metadata | 用 `cache_slots[r,257]` 判断 `p=257` 是否 hit，并在 miss 淘汰时原址更新映射 |
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

`[B,1,16384]` 中间的 `1` 是 LIDU 固定输出 contract 中的单选择列表轴，不是 sequence length，也不是 `N_idx=32/64`。当前 LIDU 没有 `S` 轴：其 query 是 `[B,N_idx,128]`，一行只代表一个请求的一个 Decode query。开启 MTP 后，vLLM 会进入 `SpecDecoding`/multi-token layout，而当前 DSA sparse path 在进入 LIDU 前就报错，不会把该 `1` 扩成 speculative token 数。

### 4.3 `cache_slots` 如何从 Top-K 的 `p` 判断 resident hit/miss

`cache_slots` 不是 LIDU 内部的私有状态，而是 `DSAResidentTokenPool` 分配并跨 Decode step 保留的 NPU tensor；LIDU 是正常 Decode 中唯一修改 token→slot 映射的算子：

```text
_cache_slots : INT32 [num_layers, max_reqs+1, W]

执行 Layer L 时：
cache_slots = _cache_slots[L]       # [pool,W]，作为 mutable 参数传给 LIDU

cache_slots[r,p]     = original position p → resident slot s
cache_slots[r,W-1]   = 当前请求、当前层的 budget metadata
```

`b` 只是当前 batch 的临时请求行；`req_pool_entries[b]=r` 将它映射到该请求稳定的 pool row，`cache_slots` 用 `r`，而本轮 HBM/DRAM block table 仍用 `b`。

对于 Top-K 选出的一个 original position `p`，LIDU 直接查询：

```text
r = req_pool_entries[b]
LIDU 查询 cache_slots[r,p]
│
├─ cache_slots[r,p] = s
│    → hit，继续使用 resident slot s，不搬 KV
│
└─ cache_slots[r,p] = -1
     → miss
     → 找到一个不属于本轮 Top-K 的 old_p
     → 复用 old_p 的 resident slot s
     → 原址更新：
          cache_slots[r,old_p] = -1
          cache_slots[r,p]     = s
     → 输出 miss pair (p,s)
```

例如 `req_pool_entries[2]=5`、Top-K 选中 `p=257`：

```text
cache_slots[5,257] = 33 → hit，LIDU 输出 topk_slots=33
cache_slots[5,257] = -1 → miss，LIDU 为它选择 slot s 并输出 (257,s)
```

LIDU 会把 miss 整理到输出前缀，并给出 `miss_count`。后续流程是：

```text
LIDU 输出 (p,s)
→ KSC 用 p 查 DRAM block table，从 DRAM 读取该 token KV
→ KSC 用 s 查 HBM block table，将 KV 写入最终 resident slot s
→ SFA-Offload 使用 s 读取
```

因此不是 KSC 搬完以后再通知 LIDU 更新；当前顺序是 **LIDU 先决定并写入映射，KSC 再填充真实 KV，SFA 最后使用**。同一 NPU stream 的顺序保证映射与 payload 一致。

`cache_slots` 中的 miss 只表示 `p` 不在 sparse resident set；仍在 HBM 的 dense tail 由 `tail_info` 旁路，不查询这张表。完整 block 的 HBM→DRAM dump 也不修改 `cache_slots`，因为 dump/block table 管历史 KV 在 DRAM 的位置，而 `cache_slots` 只管 token→resident slot。

正常 Decode 中只有 LIDU 原址更新 `cache_slots`；Python resident pool 负责初始化、request release/preempt 时清理。KSC 和 SFA-Offload 都不接收这张表。若未来把 KSC 改成异步执行，则需要额外的 `pending→ready` 状态或 device event。

源码依据：分配、清理和逐层 view 见 `dsa_resident_pool.py:25-193`、`dsa_sparse.py:502-522`；`b→r` 与查表见 `lightning_indexer_decode_update_kernel.h:197-253`；miss 淘汰及原址更新见 `lightning_indexer_decode_update_service_vector.h:591-650`；KSC 参数中没有 `cache_slots`，见 `dsa_ascend_ops_backend.py:176-204`。

### 4.4 当前 GLM 是否实现 hit 计算与 miss 传输重叠

**结论：没有。** 当前实现是三个独立算子在同一个 current NPU stream 上有序执行：

```text
LIDU 完成
  │ 输出 topk_slots + miss prefix，并原址更新 cache_slots
  ▼
KSC 完成全部 [0,miss_count) 的 DRAM→最终 resident HBM slot copy
  ▼
SFA-Offload 启动，一次性读取前 2048 个 sparse slots 和 dense tail
```

判断依据不是 Python 调用“看起来连续”，而是接口本身已经排除了 SFA 内搬 miss 的可能：

1. `execute_decode_selection_pipeline()` 在返回 selection 前依次调用 LIDU、KSC；之后 `sfa_v1.py` 才调用 SFA-Offload。
2. SFA-Offload ABI 只有 resident `key/key_rope`、`sparse_indices=topk_slots`、`tail_info` 和 HBM `block_table`；没有 DRAM arena、DRAM block table、`topk_index` 或 `miss_count`，所以它既不知道谁 miss，也不能从 DRAM 取回 miss。
3. KSC kernel 自己读取 miss prefix，对每个 `(p,s)` 完成 DRAM→UB→HBM；SFA 只在之后从 HBM 读。
4. 当前代码明确说明 cache write/dump/后续 LIDU-KSC 依赖同一 stream ordering；没有为 KSC 和 SFA 建立两个 stream/event 的并行协议。

SFA-Offload kernel 内部确实会流水搬运 HBM KV tile 到 L1 并进行 cube/vector 计算，也存在内部 stage/split-K 计算结构；那是 **SFA 自身的 HBM-read/compute pipeline**，不是“hit SFA 与 KSC miss DRAM→HBM”的跨算子重叠。

当前关键路径因此是：

```text
LIDU latency + KSC miss materialization latency + SFA-Offload latency
```

hit 不产生 KSC payload copy；但只要该 row 有 miss，单次 SFA 仍要等 KSC 的全部 miss 完成。满 block dump 在 attention 后发射，不阻塞当前 SFA 的输入准备。

你提出的优化可以研究，但属于新设计而非现状。优先方案不是额外 staging buffer，而是：

```text
copy stream：KSC 异步直接写 LIDU 已分配的最终 resident slots
compute stream：先对 hit slots 计算局部 attention 统计量
event wait 后：计算 miss slots
最后用 online-softmax 合并两部分的 max / exp-sum / weighted-value
```

不能把两次普通 SFA 的最终输出直接相加，因为 hit/miss 必须共享同一个 softmax 归一化。现有 SFA ABI 只返回最终 output，没有暴露可合并的 `max/sum/acc`，所以需要新增 split/merge 算子能力。若使用独立 staging buffer，还要让 SFA 同时寻址 resident cache 与 buffer，并为 `cache_slots` 增加 pending→ready 提交协议；计算后再搬入最终 slot 会额外增加一次 HBM→HBM copy，通常不如 KSC 直接异步写最终 slot。

源码依据：调用顺序见 `vllm-glm/vllm_ascend/dsa_sparse/dsa_sparse.py:480-565` 和 `vllm-glm/vllm_ascend/attention/sfa_v1.py:1119-1153,1226-1245`；SFA-Offload 完整 ABI 见 `sparse_flash_attention_for_offload_torch_adpt.h:19-106`；KSC copy 见 `kvcache_scatter_copy_kernel.h:52-78,91-140`；单 stream 约束见 `dsa_sparse.py:617-631`。

### 4.5 KSC 的精确地址公式

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

### 4.6 具体例子：`token_id=257` 是 miss

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

### 4.7 DENSE 与 tail 特殊路径

#### DENSE row

LIDU 的 `WriteNonOffloadedOutput()` 把前 2048 项 original positions 同时写进 `topk_index` 与 `topk_slots`，并令 `miss_count=0`。因此 KSC no-op，SFA-Offload 把这些值解释为完整 HBM cache 中的 original positions。也就是说，当前分支的 DENSE row 在地址语义上退化为原生路径。

#### dense tail

SFA-Offload 不要求把 tail 每个 slot 重复写入 16384 项数组。它读取：

```text
tail_info[b] = [tail_slot_start, tail_token_count]
real_tail_slot = tail_slot_start + tail_offset
```

再用同一套 `slot // 128`、`slot % 128`、HBM block table 公式读取连续 resident tail。

### 4.8 代码证据

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
