是每一个 Target layer 的 forward 最多调用两个 SFA，不是整个模型 forward 总共两个。

假设 Target 模型有 `L` 层，那么每层都会重复这一过程，所以最多是 `2 × L` 次 SFA。

每个 Target layer 内：

```plain
所有没有 MTP 的请求
    → 合并为 ordinary cohort
    → 调用一次普通 SFA

所有有 MTP 的请求
    → 把所有待验证 token 紧密排列
    → 调用一次 packed MTP SFA
```

如果 batch 中只有一种请求：

```plain
全是普通请求 → 只调用一次普通 SFA
全是 MTP 请求 → 只调用一次 packed MTP SFA
混合请求     → 调用两次 SFA
```

## 一个具体例子
当前 batch 有三个请求：

```plain
请求 A：没有 MTP，只有 1 个 Target token

请求 B：有 MTP，需要验证
        [yB, dB1, dB2, dB3]

请求 C：有 MTP，需要验证
        [yC, dC1]
```

框架先分组：

```plain
普通组：
[A]

MTP 组：
[yB, dB1, dB2, dB3, yC, dC1]
```

MTP 组确实会紧密排列成：

```plain
packed query =
[
  yB, dB1, dB2, dB3,
  yC, dC1
]
```

总共有 6 个待验证位置，因此：

```plain
T = 6
```

## 怎么告诉 SFA 哪些 token 属于哪个请求
框架会传递累计边界：

```plain
actual_seq_lengths_query = [4, 6]
```

含义是：

```plain
[0,4) 属于请求 B
[4,6) 属于请求 C
```

也就是：

```plain
位置 0、1、2、3 → 请求 B
位置 4、5       → 请求 C
```

因此 SFA 不会把：

```plain
请求 B 的 dB3
```

错误地当成：

```plain
请求 C 的历史 token
```

## 还需要哪些信息
除了请求边界，还需要三类信息。

第一类是每个验证位置的 HBM KV 位置：

```plain
attention_slots: [6,1,2048+tail]
```

可以理解为：

```plain
yB  应该读取哪些 HBM KV
dB1 应该读取哪些 HBM KV
dB2 应该读取哪些 HBM KV
dB3 应该读取哪些 HBM KV
yC  应该读取哪些 HBM KV
dC1 应该读取哪些 HBM KV
```

每个待验证 token 都有自己的一行 `attention_slots`。

第二类是每个请求最终的 KV 长度：

```plain
resident_seq_lengths = [B_final_length, C_final_length]
```

SFA 根据：

```plain
请求边界
+
请求最终 KV 长度
```

计算每个验证位置能看到多长的历史，保证不能看见未来 token。

第三类是每个请求的 HBM block table：

```plain
resident_block_table:
    第 0 行 → 请求 B 的 Target resident cache
    第 1 行 → 请求 C 的 Target resident cache
```

它告诉 SFA：这个请求的逻辑 KV 位置实际对应哪个 HBM block。

## 完整过程
```plain
原始 batch
[
  A,
  yB, dB1, dB2, dB3,
  yC, dC1
]
        │
        ▼
按照每个请求的 query 数分组
        │
        ├─ 普通组：[A]
        │
        └─ MTP 组：
           [yB,dB1,dB2,dB3,yC,dC1]
                  │
                  ├─ 请求边界：[4,6]
                  ├─ 每个 token 的 attention_slots
                  ├─ 请求最终长度：[B长度,C长度]
                  └─ 请求 block table：[B行,C行]
        │
        ▼
普通组调用一次普通 SFA
MTP 组调用一次 packed SFA
        │
        ▼
恢复为原始 token 顺序
[
  A_output,
  yB_output, dB1_output, dB2_output, dB3_output,
  yC_output, dC1_output
]
```

