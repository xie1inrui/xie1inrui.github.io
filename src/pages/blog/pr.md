当前确认：`mtp` 仍为 `2138e2b`，未改写；三组未跟踪文档均保留。本轮只确认冲突设计，不提交、不 push。

合并关系：

+ 集成基线：`CacheOS/vLLM-DSA:vllm-dsa-mtp`
+ 合入来源：`xie1inrui/vLLM-DSA:mtp`
+ Git 明确报告 6 个冲突文件。

### 核心取舍
上游图模式的固定地址、预热和 capture 管理继续保留，但 BatchGet ABI 全面使用 MTP 分支的新版接口：

```plain
address_table
valid_table
indices
offsets
total_miss_count int32[1]
output_indices
output
stream
synchronize
```

不恢复 `mask`。`total_miss_count` 使用固定设备 buffer，由算子在设备侧生成，热路径不调用 `.item()`、`.tolist()` 或显式读取。注释 1

### 六个文本冲突
| 文件 | 冲突内容 | 建议合并方式 |
| --- | --- | --- |
| `vllm_ascend/dsa_offload/kvaf_adapter.py` | 上游图模式预热、固定输出地址、BatchPut 同步，与 MTP 新 BatchGet ABI、调试和重复 Put 防护冲突 | 保留两边能力；BatchGet 统一换成 `total_miss_count`<br/>；BatchPut 只调用一次 KVAF，保留发布前同步和 MTP 诊断 |
| `vllm_ascend/dsa_offload/runtime.py` | 上游在 graph capture 时预热 KVAF 输出；MTP 分支安装固定 `miss_slots`<br/>、bitmap 和 profiling buffer | 两段都执行；先准备固定普通 LIDU 元数据，再预热 resident cache 输出并注册图模式 bucket |
| `vllm_ascend/dsa_offload/ops_nanovllm.py` | 上游通过 `a5_extension`<br/> 加载主 C8 算子；MTP 分支增加 `require_mtp`<br/> 和独立 LIDU extension | 保留主 C8 loader，同时保留普通/MTP 两个独立 Torch extension 边界 |
| `vllm_ascend/envs.py` | MTP 分支增加调试环境变量，上游同位置增加图模式配置 | 两边全部保留，避免覆盖 |
| `tests/ut/dsa_offload/test_kvaf_batch_get_adapter.py` | 上游是图模式预热测试；MTP 是新版 scalar ABI 测试 | 两组测试都保留；上游测试中的 `mask`<br/> 断言改为 `total_miss_count` |
| `tests/ut/dsa_offload/test_runtime.py` | 上游 fake adapter 维护图模式预热状态；MTP fake adapter 维护 BatchPut 生命周期 | 合并两个 fake 状态；buffer 统一为 scalar-count ABI，不再提供 `mask` |


### 需要额外处理的隐藏语义冲突
这些地方 Git 不一定标出冲突，但必须主动修正。

1. 图模式 KVAF 预热

上游目前用“全零 mask”发起一次空 BatchGet，目的是提前注册固定输出地址。新版 ABI 应改成：

```plain
indices/offsets/output_indices 清零
total_miss_count = 0
调用一次 KVAF BatchGet 完成输出地址预热
```

预热目的不变，只替换空请求的表达方式。

2. 图模式固定地址 key

上游 graph key 当前包含 `mask.data_ptr()`，应替换为：

```plain
total_miss_count.data_ptr()
```

这样 capture 和 replay 会校验同一组固定 scalar ABI 地址。

3. BatchPut 同步

上游要求 BatchPut 完成后同步，再向 BatchGet 发布地址；MTP 分支增加了异常诊断和重复 key 防护。合并后应当：

```plain
重复坐标检查
→ KVAF BatchPut，仅调用一次
→ 完成/异常诊断
→ 发布前同步
→ 标记该层完成
```

调试等级 2 已同步时不能重复同步。

4. 图模式与 MTP 的边界

本次“集成到图模式版本”不代表开放 MTP graph：

+ 普通 Target DSA：继续支持上游 graph capture。
+ MTP Target verification：继续只支持 eager。
+ MTP 请求进入 graph 时继续 fail fast。
+ 不把 packed MTP metadata、MTP LIDU 或 MTP SFA 捕获进图。
5. Target cache 与 draft cache

上游图模式预热只能遍历 Target resident cache。新增的 MTP draft layer：

+ 继续拥有独立 draft KV cache；
+ 不计入 Target DSA/KVAF layer 数量；
+ 不参与 Target resident cache 的图模式预热；
+ 只负责产生 draft token，验证仍由 Target 层完成。

### 已冻结内容
以下目录没有 Git 冲突，集成时保持目标分支版本不变：

```plain
csrc/attention/nanovllm_a5_ops/a5_lidu_fused_batch_get_prepare/
csrc/attention/nanovllm_a5_ops/a5_lidu_fused_batch_get_prepare_mtp/
```

不修改算子源码、Torch schema、`ops.json` 或构建脚本，也不引入 KVAF 仓库内容。

### 建议实施顺序
```plain
从 upstream/vllm-dsa-mtp 创建新集成分支
→ 合入 origin/mtp
→ 按上述设计解决 6 个冲突
→ 修正 graph prewarm/key 的隐藏 mask 引用
→ 审查普通 graph 与 MTP eager 的隔离
→ 静态检查
→ 提交并 push 新分支
→ 创建无冲突 PR
```

如果这套取舍得到你和 mentor 的确认，我下一步就按此方案重新整理集成 worktree 并实施

