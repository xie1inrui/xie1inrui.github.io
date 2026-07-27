# 分布式 KV Pool

> 状态：第一版架构与协议可行性草案，不代表当前 MemCache 已经实现。  
> 日期：2026-07-24  
> 阅读方式：先固定一个简单 DP 场景，随后立刻沿 Get、Put、offload、Remove 和 slot 复用的时间线观察整个系统；组件和一致性原理放在流程之后解释。  
> 边界：DSA 只提供“已经选出的 KV keys”；数据搬运层只作为黑盒，不展开 DSA 算法和 MemFabric/HComm/硬件内部。

---

## 0. 先说结论

完全分布式可以做到，但“完全分布式”不等于“完全没有协调”：

~~~text
稳态Decode Get：不访问Owner，不做远程元数据同步
本地capability miss时：按key访问对应Owner Shard完成地址解析和授权
Put/offload/remove：只在对应Owner Shard内形成唯一顺序
物理slot分配和回收：只由所在rank的Local Pool Manager管理
任何状态无法证明安全：miss/重算，不把可疑KV交给Attention
~~~

本方案组合五种机制：

1. **哈希分片**：任何 rank 都能由 key 算出负责它的 Owner Shard，不需要全局 key 总表；
2. **分片共识**：只为 Put、迁移和 Remove 等低频变化建立唯一顺序；
3. **不可变对象与 Copy-on-Write**：已经发布的 KV 不原地改写；
4. **Capability/Lease**：Decode Agent 提前取得一段时间有效的直接读取权限；
5. **延迟回收**：先停止新读，再等旧读结束，最后复用物理 slot。

哈希分区可借鉴 Dynamo 的一致性哈希思想，但本方案不采用 Dynamo 的最终一致冲突语义；每个 Owner Shard 的元数据变化由复制状态机串行化。[Amazon Dynamo 论文](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)

Raft、Lease 和 RCU 分别为元数据复制、无逐读查询的缓存安全以及延迟回收提供成熟基础：[Raft](https://raft.github.io/raft.pdf)、[Leases](https://web.stanford.edu/class/cs240/readings/leases.pdf)、[Linux RCU](https://docs.kernel.org/6.8/RCU/Design/Requirements/Requirements.html)。

本方案保证：

> Attention 只消费仍受有效 capability 保护、已经 Commit 且身份校验一致的 KV；无法证明这些条件时返回 miss/重算。

本方案允许目录短暂陈旧，但不允许数据错误：

- Get 不一定立刻知道刚新增的更快副本；
- 网络分区时 Put、Remove 和迁移可能暂停；
- 并发 Put 产生新版本时，已获准的快照读者可以继续读取旧的已提交版本。

### 0.1 执行这个方案的预计内存代价

以下按一个rank对应一张NPU卡、每rank约32 GiB HBM KV Pool估算：

| 代价 | 每rank预计需要 |
|---|---:|
| BlockHeader等必需HBM元数据 | 小于0.01 GiB，基本可以忽略 |
| 预取目标slot、在途迁移缓冲 | 建议预留0.25～0.5 GiB HBM |
| Owner、SlotRecord、capability、订阅和日志 | 建议预留0.5 GiB Host DRAM |

因此工程上可以先按下面的数字准备：

~~~text
每rank额外HBM预算：      0.25～0.5 GiB
每rank额外Host DRAM预算：约0.5 GiB
~~~

如果预取slot直接从已有KV Pool中分配，就不需要物理增加HBM，只是32 GiB中会有0.25～0.5 GiB不能用于长期驻留KV。

最大的容量代价不是元数据，而是KV副本。例如所有KV都长期保存两份副本时，256 GiB物理Pool只能保存约128 GiB唯一KV；实际应只给重要或高复用KV保留双副本。

### 0.2 执行这个方案的预计通信代价

| 场景 | 新方案预计代价 | 与当前约10 ms MetaService相比 |
|---|---|---|
| 稳态Get，capability命中 | 本地查表；远程元数据RPC为0 | 能去掉一次约10～12 ms的Meta查询，payload搬运时间不变 |
| 首次Get或capability miss | 约2～4段Owner/Store控制消息 | 优化目标约1～5 ms，但尚未实测，不保证一定更快 |
| Put、Offload、Remove | 多次定向通知、ACK和Owner日志复制 | 通信次数更多，但放在后台，不应阻塞普通Decode Get |

现有4 MiB A3/device_sdma样本中，同步Meta BatchGet平均约12.27 ms；该数字只作为量级参考，不代表A5或真实DSA workload的固定结果。[性能拆解](./MemCache_and_MemFabric_性能拆解_device_sdma.md)

所以最终判断是：

> 稳态Decode会更快，因为它不执行多步远程协调；多步通信只发生在首次解析和KV状态变化时，并通过提前准备、批量和异步执行移出Decode关键路径。

---

## 1. 贯穿全文的统一场景

### 1.1 并行和部署假设

第一版只讨论最简单的纯 DP：
~~~text
DP = 4
TP = 1
PP = 1
EP = 1
~~~
四个 DP rank 都是完整 vLLM Engine，都可以独立执行 Prefill 和 Decode。当前快照只展开 rank 2 正在服务请求 R1；rank 0、rank 1、rank 3 仍可同时服务各自的其他请求，不是专用存储节点。
~~~text
机器A
├── DP rank 0 / NPU0
│   ├── 可独立Prefill和Decode
│   ├── HBM Pool：S17/G42保存K7 V3
│   └── PeerService 0
└── DP rank 1 / NPU1
    ├── 可独立Prefill和Decode
    ├── Host DRAM Pool：D81/G9保存K7 V3
    └── PeerService 1

机器B
├── DP rank 2 / NPU2
│   ├── 当前正在Decode长上下文请求R1
│   ├── R1本地HBM已有K1、K2
│   └── PeerService 2 / Decode Agent R1
└── DP rank 3 / NPU3
    ├── 可独立Prefill和Decode
    ├── 贡献HBM、Host DRAM、SSD Pool
    └── PeerService 3
~~~

### 1.2 先认识图里会反复出现的概念

| 概念 | 在本文中的含义 | 本例 |
|---|---|---|
| rank | 一个参与推理和 KV Pool 的服务身份；简单 DP 中近似一个 DP Engine/NPU | rank 0～3 |
| slot | Local Pool Manager 可独立 Reserve、Commit、Retire、Free 的本地物理空间 | HBM S17、DRAM D81 |
| object version | 逻辑 KV 内容版本 | K7 V3 |
| generation | slot 每次回收再利用时递增的物理代数 | S17/G42，复用后变 G43 |
| PeerService | 每个 rank Host CPU 上的 KV Pool 控制服务 | PeerService 0～3 |
| Owner Shard | 管一批 keys 的版本、有效副本集合和变更顺序 | shard 27 管 K7 |
| RPC | 一个 rank 向另一个 Host CPU 服务发送小型控制请求并取得响应 | rank2向shard27准备capability |
| descriptor | 描述一个候选物理副本的位置和身份 | rank0/HBM/S17/G42 |
| capability | 允许特定 holder 在有效期内直接读取某个 descriptor 的凭证（允许别人读期间，该rank保证这个数据不会被修改） | cap-A、cap-B |
| lease | capability 的有效期限；期限内 Store 必须保护源 slot | expires=T100 |
| 数据搬运层 | 把源 slot payload 搬到目标 slot 的黑盒 | 只观察成功或失败 |

必须区分三种编号：

~~~text
K7 V3：K7逻辑内容的第3个版本
S17/G42：物理slot S17当前处于第42代
rank0/inc17：rank0当前是第17次进程/节点生命周期
~~~

同一个 K7 V3 的两个副本可以拥有完全不同的 slot generation。

### 1.3 K7 的元数据和 payload 分别在哪里

~~~text
OwnerShard(K7) = shard 27

shard 27元数据复制组：
├── leader：rank 3 Host CPU
├── follower：rank 0 Host CPU
└── follower：rank 2 Host CPU

K7 V3已提交payload副本：
├── replica A：rank 0 / HBM / S17 / generation 42 / cap-A
└── replica B：rank 1 / DRAM / D81 / generation 9 / cap-B
~~~

两组“副本”不是同一件事：

~~~text
shard 27复制组
    保存小型元数据账本：K7哪个版本有效、有哪些副本、谁在订阅

K7物理副本
    保存Attention真正需要的KV payload
~~~

因此 rank 3 可以管理 K7 的逻辑状态而不保存 K7 payload；rank 1 可以保存 K7 payload 而不是 shard 27 follower；rank 0 则同时承担两种独立角色。

### 1.4 K7 为什么从 rank 2 跑到了 rank 0 和 rank 1

本文设定 K7 是 R1 自己的较老历史 KV，而不是必须依赖跨用户 Prefix 复用：

~~~text
T0  R1在rank2完成Prefill，生成K1～K9
T1  R1上下文增长，rank2 HBM压力上升
T2  K7封存为不可变K7 V3
T3  Pool在rank0 HBM创建replica A
T4  Pool在rank1 DRAM创建replica B
T5  两个副本Commit后，rank2释放自己的旧K7副本
T6  R1继续在rank2 Decode，DSA后来再次选中K7
~~~

第一版只共享已经封存的完整 block。若 DSA 四 token 为一组，则不足四 token 的尾部 block 保持请求私有和 `WRITING`，组完整后才能 Commit 到 Pool。

### 1.5 本文沿着哪些事件证明方案可行

| 事件 | 输入 | 期望结果 |
|---|---|---|
| 地址解析与授权 | `BatchGet`输入K7且本地capability miss | rank2取得并缓存cap-A、cap-B |
| 稳态 Get | DSA已经选出K1、K2、K7 | K1/K2本地使用，K7不问Owner直接读取 |
| Put | rank2产生并封存K10 | K10写完整后才对其他rank可见 |
| Offload/撤销 | rank0要回收K7的HBM S17 | rank2切换到rank1/D81，旧读仍安全 |
| Remove | 删除K7 V3 | 停止新Resolve，旧读按语义排空 |
| slot复用 | S17需要保存K20 | 只有K7保护结束后才能从G42变G43 |

---

## 2. 在统一场景中看系统如何运转

### 2.0 如何阅读本章流程图

本章只画 KV Pool 边界内的参与方：

~~~text
输入边界：DSA已经给出所需keys，或上层已经产生封存payload
控制面：Decode Agent、Owner Shard、Local Pool Manager
数据面：数据搬运层黑盒
输出边界：本地READY，或Put/Remove返回状态
~~~

图中的实线请求分三类：

| 类型 | 例子 | 是否搬 payload |
|---|---|---:|
| 本地操作 | capability查表、Header校验、READY | 否 |
| RPC控制消息 | Resolve、Reserve、CommitAck、Revoke | 否 |
| 数据搬运 | rank0/S17 → rank2/P30 | 是，但内部不展开 |

#### 全章参与方总览

~~~text
rank2 Decode Agent
    管R1的请求级快速目录和READY状态

shard27 Owner复制组
    管K7的逻辑版本、副本集合、订阅者和变更顺序

rank0/rank1 Local Pool Manager
    管各自物理slot以及何时可以安全回收

数据搬运层（黑盒）
    只接受已验证的source、destination、size，返回完成或失败
~~~

### 2.1 地址解析与读取授权：只取得 K7 的位置和 capability，本图不搬 KV 数据

第 2.1 和第 2.2 节不是两个对上层可见的独立请求，而是同一次 `BatchGet(keys)` 在 KV Pool 内部的两个阶段：

~~~text
KV Pool收到BatchGet([K1,K2,K7])
→ 本地CapabilityLookup
→ 已命中的key直接进入第2.2节
→ 未命中的key执行第2.1节地址解析与授权
→ 全部目标key具备可用capability后进入第2.2节搬运
~~~

KV Pool 不区分 key 在 Top-K 前还是之后产生，只区分“本地 capability 命中”与“本地 capability 缺失”。

#### 2.1.1 开始前状态

~~~text
KV Pool收到key=K7
Owner已经记录K7 V3的两个COMMITTED副本
rank2本地Capability Cache尚无K7 descriptor/capability
~~~

#### 2.1.2 详细时序图

~~~mermaid
sequenceDiagram
    autonumber
    participant A as rank2：K7的使用者<br/>Decode Agent R1
    participant O as rank3：K7目录负责人<br/>shard27 leader
    participant R as 目录备份<br/>shard27 followers
    participant S0 as rank0：保存K7的HBM副本<br/>S17/G42
    participant S1 as rank1：保存K7的DRAM副本<br/>D81/G9

    Note over A,S1: 这是“查地址并取得读取许可”的准备图；整张图只传小型元数据，不传K7 payload
    Note over A,S1: KV Pool只感知“输入key=K7且本地没有capability”；不感知这个key在Top-K之前还是之后产生
    A->>A: ComputeOwnerShard(K7)=shard27<br/>本地哈希定位目录负责人；此时还不知道物理地址
    A->>O: RPC BatchPrepareCapabilities(R1,[K7])<br/>查询K7副本，并为rank2申请直读凭证
    O->>O: Lookup ObjectRecord(K7)<br/>账本显示K7 V3=READABLE，rank0 HBM和rank1 DRAM各有一份
    O->>R: Replicate LeaseGrant(owner_epoch=61,holder=rank2,expiry=T100)<br/>把“rank2可读到T100”写入shard27复制日志
    R-->>O: Quorum ACK<br/>多数派已保存；leader故障后新leader也不会忘记这次授权
    O->>S0: RPC ProtectReplica(K7,V3,S17,G42,T100,holder=rank2)<br/>确认rank0副本，并要求T100前不得复用S17
    S0->>S0: Validate BlockHeader + Install Protection<br/>核对K7/V3/G42/COMMITTED，登记holder和protect_until
    S0-->>O: ReplicaCapability cap-A READY<br/>rank2可在T100前直读rank0/HBM/S17/G42
    O->>S1: RPC ProtectReplica(K7,V3,D81,G9,T100,holder=rank2)<br/>确认rank1副本，并要求T100前不得复用D81
    S1->>S1: Validate BlockHeader + Install Protection<br/>核对K7/V3/G9/COMMITTED，登记holder和protect_until
    S1-->>O: ReplicaCapability cap-B READY<br/>rank2可在T100前直读rank1/DRAM/D81/G9
    O-->>A: ResolveResponse{K7 V3,cap-A,cap-B,update_seq=104}<br/>返回两个安全候选及其读取凭证
    A->>A: Atomic Install into R1 Capability Cache<br/>把地址和许可缓存到本地；后续选中K7时不再询问Owner
    Note over A,S1: 解析结束：K7仍在rank0 S17和rank1 D81；真正搬运发生在第2.2节
~~~

#### 2.1.3 每一步发生了什么

| 图中步骤 | 专业动作 | 大白话 | 状态变化与意义 |
|---:|---|---|---|
| 1 | `ComputeOwnerShard` | 用Hash算出K7归shard27管 | 只定位Owner，不定位payload |
| 2 | `BatchPrepareCapabilities` RPC | 问Owner“K7在哪，并给我读取许可” | 只传key和请求信息，不传K7 payload |
| 3 | `Lookup ObjectRecord` | 查目录账本，找到两个已发布副本 | 目录记录有效，不等于物理slot已经完成本次保护 |
| 4～5 | `LeaseGrant`复制并取得`Quorum ACK` | 多数派记住rank2可以读到T100 | leader切换后仍能恢复已有授权 |
| 6 | `ProtectReplica` RPC | 要求rank0确认S17/G42并暂时不要复用 | 这是控制请求，不读取K7 payload |
| 7 | `Validate BlockHeader`并安装保护 | rank0核对slot身份，登记holder和截止时间 | S17进入受保护状态 |
| 8 | 生成`ReplicaCapability cap-A` | 给rank2一张限时读取rank0副本的许可证 | cap-A只对应S17/G42这一份副本 |
| 9～11 | 对rank1执行`ProtectReplica → Validate → cap-B` | 同样确认并保护D81/G9 | 得到独立备用许可证cap-B |
| 12 | `ResolveResponse` | Owner把两个地址、两张许可证和目录序号返回rank2 | 仍然没有搬运K7 payload |
| 13 | `Atomic Install Capability Cache` | rank2把结果一次性装入R1本地表 | 后续稳态Get本地选cap-A或cap-B，不问Owner |

#### 2.1.4 结束后状态

~~~text
R1 Capability Cache
└── K7 V3
    ├── cap-A → rank0/HBM/S17/G42/expires=T100
    └── cap-B → rank1/DRAM/D81/G9/expires=T100
~~~

只要本地 `CapabilityLookup(K7)` miss，KV Pool 就执行本节协议。上层可以选择提前调用以隐藏 RPC，也可以在提交 keys 时调用；这是调用时机优化，不改变 KV Pool 内部协议，也不需要拆成两个场景。

~~~text
本图所有网络消息都是小型控制信息：
key、版本、rank、slot、generation、lease、capability、update_seq。

本图传输的K7 payload字节数 = 0。
~~~

### 2.2 Get：输入 keys 为 K1、K2、K7，这里真正搬 KV 数据

#### 2.2.1 开始前状态

~~~text
输入：selected_keys = [K1, K2, K7]

rank2本地：
K1 → local HBM / READY
K2 → local HBM / READY

rank2 Capability Cache：
K7 → cap-A(rank0 HBM) + cap-B(rank1 DRAM)
~~~

#### 2.2.2 真正的数据搬运时序图

~~~mermaid
sequenceDiagram
    autonumber
    participant A as rank2 Decode Agent R1
    participant H0 as 源：rank0 HBM<br/>S17/G42
    participant X as 数据搬运层（黑盒）
    participant L as 目标：rank2 Local Pool/HBM<br/>预取slot P30

    Note over A,X: 输入边界：DSA已经选出K1、K2、K7；不展开DSA内部
    A->>A: BatchLookup([K1,K2,K7])
    A->>L: 校验K1、K2本地READY
    L-->>A: local_ready=[K1,K2]
    A->>A: 过滤K7候选：version/state/lease/incarnation
    A->>A: 成本选择：cap-A预计比cap-B更快
    A->>L: Reserve本地预取slot P30
    A->>A: 进入K7本地read epoch

    par 本地部分立即可消费
        A->>L: 返回K1、K2本地READY地址
    and 远端K7预取
        A->>X: 提交ReadByCapability(cap-A)<br/>source=S17/G42, destination=P30
        activate X
        Note over H0,L: KV payload搬运区间开始
        X->>H0: 发起源slot读取
        activate H0
        H0-->>X: 返回K7 V3 Header + payload
        deactivate H0
        X-->>L: 将K7 V3 Header + payload写入P30
        Note over H0,L: KV payload搬运区间结束；P30已有数据但尚未READY
        deactivate X
        A->>L: 校验key/version/generation/incarnation/state/checksum
        L-->>A: Header匹配
        A->>L: 原子设置READY(K7,P30)
    end
    A->>A: 退出read epoch，记录副本延迟EWMA
~~~

#### 2.2.3 每一步发生了什么

| 图中步骤 | 执行者 | 专业动作 | 大白话 | 是否在 Get 关键路径 |
|---:|---|---|---|---:|
| 1～3 | rank2 Agent/本地Pool | 批量查本地目录，确认K1/K2 READY | 先看看所需数据中哪些已经在自己卡上；K1、K2可以直接用 | 是，但全是本地操作 |
| 4 | rank2 Agent | 过滤版本、状态、lease、incarnation | 检查K7的候选地址是否仍然有效，过期或身份不对的地址直接丢掉 | 是，本地操作 |
| 5 | rank2 Agent | 在正确候选中估计完成时间 | 在安全候选里挑预计最快到达的那一份，本例选择rank0 HBM | 是，本地操作 |
| 6 | rank2 Local Pool | 为K7分配本地目标P30 | 在自己卡上先准备一个空位，用来接收远端搬回来的K7 | 是，本地allocator |
| 7 | rank2 Agent | 标记本地在途读取epoch | 登记“我正在读旧地址”，防止源slot在搬完前被回收覆盖 | 是，用于安全排空 |
| 8 | rank2 Agent | K1/K2无需远端读取即可交付 | 本地已有的K1、K2先交给计算，不必等待K7 | 是 |
| 9 | rank2 Agent | 提交cap-A、源S17/G42和目标P30 | 告诉搬运层“从rank0的S17把K7搬到rank2的P30”；此时只是下发任务 | 是；还没有payload返回 |
| 10～12 | 源HBM、搬运黑盒、目标HBM | 搬运K7 Header和payload | 这里才真正跨rank读取K7，并把数据写入rank2的P30 | 是；不访问Owner或源Store控制RPC |
| 13～15 | rank2 Agent/Pool | 校验Header并原子设置READY | 搬到本地后再核对“是不是K7 V3、是不是原来的slot代数”；全部正确才允许计算使用 | 是，错误则不能交付 |
| 16 | rank2 Agent | 退出epoch并更新性能统计 | 告诉本地“这次远端读结束了”，顺便记录耗时供下次选择副本 | 可在尾部/后台完成 |

因此按图中的自动编号：

~~~text
第9步：提交搬运描述，不是payload本身
第10步：搬运黑盒向源slot发起读取
第11步：K7 Header + payload从rank0 HBM进入搬运路径
第12步：K7 Header + payload写入rank2预取slot P30
第13步：搬运已经完成，开始做本地身份和完整性校验
~~~

#### 2.2.4 Capability已命中时，Get中没有谁

~~~text
Owner RPC          = 0
Owner quorum       = 0
远程目录查询       = 0
源Store控制RPC     = 0
全rank广播         = 0

本地capability查找 = 1个batch
远端payload搬运    = 仅K7需要
本地Header/READY   = 每个搬回block一次
~~~

读取结束后，K7/P30 至少在 R1 需要期间保持本地 READY。若将其作为跨请求长期共享副本，还必须执行正式 replica Commit；否则它只是 R1 的请求级副本。

### 2.3 多副本选择：为什么不固定 HBM 优先

对 K7，Agent 先做正确性过滤：

~~~text
候选A：rank0/HBM/S17/G42/V3/cap-A
候选B：rank1/DRAM/D81/G9/V3/cap-B

必须满足：
object_version == V3
replica_state == COMMITTED
capability未撤销且剩余时间足够
node incarnation匹配
descriptor未被更高update_seq移除
~~~

然后才比较预计完成时间：

~~~text
estimated_completion =
    queue_delay_ewma
  + transfer_bytes / effective_bandwidth_ewma
  + topology_penalty
  + staging_cost
~~~

因此远端 HBM 不一定永远比拓扑更近、队列更短的 DRAM 快。Agent 选择最小预计完成时间，并保留第二候选作为读取失败后的 fallback。

### 2.4 Put：rank 2 发布新 block K10

#### 2.4.1 开始前状态

~~~text
输入边界：rank2已经产生并封存K10 payload
K10尚未进入任何Owner的READABLE集合
OwnerShard(K10)=shard91
shard91 leader=rank0，followers=rank1/rank3
目标策略：rank2 HBM一份 + rank1 DRAM一份

本例可直接分配的安全空闲slot：
├── rank2/HBM/H20/G6，state=FREE
└── rank1/DRAM/D90/G13，state=FREE
~~~

#### 2.4.2 Reserve 会不会直接覆盖旧 block

不会。`Reserve(K10)` 只能取得 `FREE` slot，不能把仍为 `COMMITTED/RETIRING` 的旧 block 直接改写成 K10。

若 free-list 为空，allocator 必须先选择其他 rank，或者完整执行旧副本安全回收协议。下面这张图是 Put 进入写数据阶段之前的分配门禁：

**本图目标：** 为即将写入 Pool 的 K10 找到安全物理空间。本例优先申请 `rank2/HBM/H20/G6` 和 `rank1/DRAM/D90/G13`；这张图只决定“哪些 slot 可以写”，还不搬运 K10 payload。若没有 `FREE` slot，则必须先把选中的旧副本安全撤销并回收，绝不能直接覆盖它。

~~~mermaid
flowchart TD
    A["ReserveReplica(K10)<br/>为K10申请物理slot"] --> B["Local Allocator查free-list<br/>只查已经安全释放的slot"]
    B --> C{"存在FREE slot？"}
    C -->|"是"| D["FREE → RESERVED<br/>返回新descriptor，允许Put继续"]
    C -->|"否"| E{"本机其他介质或其他rank<br/>有符合放置策略的FREE slot？"}
    E -->|"是"| F["重定向Reserve到安全空位<br/>避免让Put同步等待Offload"]
    E -->|"否，或策略强制要求本卡HBM"| G["SelectVictim(old key/slot/generation)<br/>在本卡选择一个待淘汰旧副本"]
    G --> H["ProposeRetireReplica(old key)<br/>向旧key的Owner申请撤销副本"]
    H --> I["Owner提交RETIRING<br/>停止签发新的旧capability"]
    I --> J["REVOKE_CAPABILITY / REMOVE_REPLICA<br/>定向通知旧capability holders"]
    J --> K{"所有旧读取已经排空？"}
    K -->|"收到ACK"| M["确认无有效capability、无在途搬运<br/>经过grace period"]
    K -->|"holder失联"| L["等待lease expiry<br/>+ clock guard + max transfer time"]
    L --> M
    M --> N["old Header → FREE<br/>slot generation递增"]
    N --> O["FREE → RESERVED for K10<br/>此时新Put才允许写入"]
~~~

| 阶段 | 专业动作 | 大白话 | 是否通知旧读者 |
|---|---|---|---:|
| 有空闲空间 | `FreeListPop → Reserve` | 直接拿一个已经安全空闲的slot | 否，因为没有旧对象和holder |
| 换目标rank | `PlacementRetry` | 当前rank没空间，去另一个有空闲slot的rank | 否 |
| 选择旧副本 | `SelectVictim + ProposeRetireReplica` | 想回收某个旧block，但此时还不能写K10 | 尚未覆盖 |
| 停止新读 | Owner提交`RETIRING` | 不再给任何新读者发旧地址的capability | 向已有holder定向发送revoke |
| 排空旧读 | `Revoke ACK`或`Lease Expiry` | 等已经拿到旧许可证的人全部停止使用 | 是，且必须发生在覆盖之前 |
| 安全释放 | `FREE + generation++` | 旧block正式离开slot，旧地址身份失效 | 保护已经完成 |
| 新对象写入 | `RESERVED → WRITING → COMMITTED` | 现在才允许把K10写进这个slot | 不再需要通知旧读者 |

##### rank2 没有位置时：去其他位置 Put，还是先 Offload 再写回 rank2

两种都允许，但本方案给出明确的默认优先级：

~~~text
1. rank2目标介质存在FREE slot
   → 直接在rank2 Reserve并Put

2. rank2 HBM没有FREE slot，但放置策略允许换位置
   → 优先选择本机DRAM或其他rank的FREE slot
   → 直接把K10 Put到那个安全位置
   → 不让本次Put同步等待旧block Offload

3. 策略强制要求rank2必须保留一份HBM副本，
   或其他合法位置也没有FREE slot
   → 在rank2 HBM选择冷的旧副本victim
   → 必要时先为victim建立DRAM/远端替代副本
   → 撤销victim旧HBM副本并排空旧读取
   → S17等旧slot安全变为FREE且generation递增
   → 再把K10写入刚腾出的rank2 HBM slot
~~~

因此，“没有位置”不等于一定先 Offload。默认先找别处的安全空位，因为这条路径更快；只有必须获得本卡 HBM 空间时，才让 Put 等待本卡 victim 的安全回收。

这里被 Offload 的是占用空间的旧 victim，而不是刚产生的 K10。还要进一步区分：

| victim 状态 | 腾出本卡 HBM 前要做什么 |
|---|---|
| victim 已经在其他位置有可用副本 | 不必再搬 payload；确认备用副本可读后，撤销本卡 HBM 副本即可 |
| victim 是最后一份可用副本 | 必须先把 victim 搬到 DRAM 或其他 rank 并 Commit，再撤销本卡 HBM 副本 |
| victim 允许直接丢弃，后续可以 miss/重算 | Owner 先提交相应删除或无副本状态，再安全撤销本卡 HBM 副本 |

以统一环境为例：如果 K10 强制要求在 rank2 HBM 保留一份，但 rank2 HBM 已满，可以选择 rank2 上的冷 block Kx。若 Kx 已经有 DRAM 副本，就直接撤销 Kx 的本地 HBM 副本；若 Kx 只有这一份，则先执行 `Kx：rank2 HBM → DRAM/远端` 的 add-before-remove Offload。等旧 slot 安全释放后，K10 才能写进该 slot。K10 的第二份副本仍可以并行写入 `rank1/DRAM/D90/G13`。

因此不应设计成：

~~~text
先覆盖旧block → 再通知读者“数据脏了”       // 已经太晚，不安全
~~~

而必须是：

~~~text
先通知并阻止旧读 → 等旧读排空 → Free并增加generation → 再写新block
~~~

这不是“Put 修改一个正在共享的 block”，而是“Put 只写安全的新 generation”。

这里的通知可以异步发送，但物理 slot 复用必须被同步门禁挡住：只有全部相关 holder ACK，或者它们的 lease 到期并经过安全窗口，allocator 才能把旧 slot 放回 free-list。通知也不是说“已经读到本地的 K7 内容变脏了”；一个已经完整搬到其他 rank 并校验 READY 的不可变 K7 V3 仍然正确。被撤销的是“以后继续从旧物理地址读取”的权利。

如果 victim 是对象的最后一个副本，Owner 还必须先创建替代副本，或者按策略明确把对象变成无副本、后续 Get miss；Local Pool Manager 不能自行删除全局最后一份数据。

#### 2.4.3 正常有 FREE slot 时的详细 Put 时序图

**本图目标：** rank2 已经计算并封存 K10，本流程要把 K10 保存成两个可共享副本：本地副本写入 `rank2/HBM/H20/G6`，第二副本写入 `rank1/DRAM/D90/G13`。只有两份 payload 都写完整、Store 标记为 `COMMITTED`，并由 shard91 多数派发布为 `READABLE` 后，其他 rank 才能查到 K10。

~~~mermaid
sequenceDiagram
    autonumber
    participant A as rank2 Pool Publisher
    participant O as shard91 leader<br/>rank0 Host CPU
    participant R as shard91 followers<br/>rank1/rank3 Host CPU
    participant S2 as rank2 Local Pool Manager
    participant S1 as rank1 Local Pool Manager
    participant X as 数据搬运层（黑盒）

    Note over A,X: 输入边界：K10 payload已封存；H20/G6和D90/G13已经安全FREE；Put绝不覆盖COMMITTED block
    A->>A: ComputeOwnerShard(K10)=shard91
    A->>O: RPC PutStart(K10,size,replicas=2,local-first)
    O->>O: 检查不存在冲突的READABLE对象
    O->>R: 复制PREPARING(V1,mutation_token=M55,owner_epoch=12)
    R-->>O: quorum确认
    O->>S2: RPC Reserve(K10,V1,HBM,M55)
    S2->>S2: FreeListPop H20/G6<br/>state FREE→RESERVED
    S2-->>O: descriptor H20/G6
    O->>S1: RPC Reserve(K10,V1,DRAM,M55)
    S1->>S1: FreeListPop D90/G13<br/>state FREE→RESERVED
    S1-->>O: descriptor D90/G13
    O-->>A: WritePlan{rank2/H20,rank1/D90,M55}

    par 写本地副本
        A->>S2: 写Header(state=WRITING)+payload
        S2->>S2: 校验完整性，Header→COMMITTED
        S2-->>O: CommitAck(H20,G6,M55)
    and 写第二副本
        A->>X: 搬运K10：rank2 source → rank1/D90
        X-->>S1: payload完成或失败
        S1->>S1: 校验完整性，Header→COMMITTED
        S1-->>O: CommitAck(D90,G13,M55)
    end

    O->>O: 只接受当前epoch/token的CommitAck
    O->>R: 复制READABLE(V1,replicas=H20+D90)
    R-->>O: quorum确认
    O-->>A: Put成功：K10 V1已发布
~~~

#### 2.4.4 每一步发生了什么

| 图中步骤 | 阶段 | 关键状态 | 大白话 | 为什么安全 |
|---:|---|---|---|---|
| 1～2 | 找Owner并发起Put | K10还不可读 | rank2用Hash找到负责K10目录的shard91，并申请发布K10 | Hash只决定元数据shard，不代表K10已经写入Pool |
| 3～5 | Owner准备 | `PREPARING + M55`获多数派 | Owner先给这次Put发一个唯一写入令牌M55，并让多数目录副本记住它 | 并发Put获得唯一mutation顺序 |
| 6～11 | 两个Store执行`FreeListPop + Reserve` | H20/G6、D90/G13从FREE变RESERVED | rank2 HBM和rank1 DRAM各拿一个真正空闲的slot，先圈起来不让别人使用 | 只使用安全free-list，不覆盖任何旧COMMITTED block |
| 12 | 返回WritePlan | 写者得到明确目标和token | Owner告诉rank2“把本地副本写到H20、第二副本写到D90，并带上M55” | 旧leader计划会被epoch拒绝 |
| 13～19 | 写payload并Store Commit | Header从WRITING变COMMITTED | 两份数据并行写入；每个Store只有在数据完整、校验通过后才盖上“已完成”标记 | 半写对象没有CommitAck，读者看不到 |
| 20 | Owner校验Ack | epoch/token必须匹配 | Owner确认两个完成回执确实属于当前这次Put，而不是旧请求或重复消息 | 拒绝重复或失效写者 |
| 21～23 | Owner发布READABLE | replica set获多数派 | 多数目录副本确认后，Owner才正式宣布“K10现在可以被其他rank查到并读取” | 只有此后新Resolve才能看到K10 |

Put 的关键线性化边界是：

~~~text
payload写完 ≠ 全局可见
Store COMMITTED ≠ Owner已经发布
Owner quorum提交READABLE = 新读者可以发现K10
~~~

两个请求并发 Put 相同 K10 时，不需要锁住整个 Pool；只有 shard91 leader 为该 key 串行授予 mutation token。若 key 是内容哈希且 payload 相同，后来的 Put 可以返回 duplicate/已存在。

#### 2.4.5 结束后状态

~~~text
OwnerShard(K10)：K10 V1 = READABLE
├── rank2/HBM/H20/G6/COMMITTED
└── rank1/DRAM/D90/G13/COMMITTED

任何其他rank第一次Resolve K10时，只会看到这两个已经Commit的副本。
~~~

### 2.5 完整Offload：先复制新副本并通知读者，再释放旧HBM副本

#### 2.5.1 主流程：K7从rank0 HBM切换到DRAM

通用规则是 add-before-remove：先确保目标 DRAM 副本已经 `COMMITTED` 并可被读者使用，再撤销旧 HBM 副本。

**本图目标：** 把 K7 V3 从即将回收的 `rank0/HBM/S17/G42` Offload 到新的 `rank3/DRAM/D44/G7`。这里的 Offload 不是一次具有破坏性的“搬走”：系统先执行 `S17 → D44` 的只读复制，同时继续保护并保留旧 S17；等 D44 发布、rank2 确认安装新 capability 后，才通知 rank2 停止从 S17 发起新读；所有已经开始的旧读排空后，S17 才能真正释放。

因此必须区分三个时刻：

~~~text
新副本复制完成：D44已有完整K7，但旧S17仍然可读，不能释放
读者切换完成：rank2已安装新capability，并停止从旧S17发起新读
Offload物理完成：旧read排空，S17经过grace period后才变为FREE
~~~

如果实现成“复制完成后立刻释放或覆盖 S17，再通知 rank2”，确实会造成错误；rank2 可能仍凭 cap-A 读取已经变化的旧地址。本方案明确禁止这种顺序。

统一环境中其实已经有 `rank1/DRAM/D81/G9` 这份可用副本。因此，本图故意画的是“原先没有可依赖 DRAM 副本，必须新建一个副本”的完整版本；统一环境实际执行时可以采用下面的快速路径。

> 若 rank1/D81 已经存在可用的 K7 V3 且 rank2 已持有 cap-B，则下面创建并安装新副本的第 1～11 步可以跳过，只执行“确认 D81 和 cap-B 仍有效 → 通知并撤销旧 HBM → 排空旧读 → 释放 S17”的后半段。

~~~mermaid
sequenceDiagram
    autonumber
    participant O as shard27 Owner
    participant S0 as rank0 Local Pool Manager
    participant N as 新目标Local Pool Manager
    participant X as 数据搬运层（黑盒）
    participant A as rank2 Decode Agent

    O->>N: ReserveReplica(K7,V3,target=rank3 DRAM)
    N-->>O: 新slot D44/G7，state=RESERVED
    O->>S0: ProtectSourceForMigration(S17,G42)<br/>为复制增加保护，禁止释放或覆盖S17
    Note over S0,A: 复制阶段：K7不可变，S17仍为COMMITTED且受保护；rank2可继续凭cap-A安全读取
    O->>X: 只读复制K7：S17/G42 → rank3/D44/G7<br/>复制不删除旧S17
    X-->>N: payload完成或失败
    N->>N: 校验Header/payload，D44/G7→COMMITTED
    N-->>O: CommitAck(new replica)
    O->>O: quorum发布ADD_REPLICA(rank3,D44,G7)
    O-->>A: ADD_REPLICA + new capability
    A->>A: 先安装新capability
    A-->>O: ADD_ACK(new capability installed)
    O->>O: quorum提交旧S17/G42→RETIRING
    O->>S0: BEGIN_RETIRE(S17,G42,holders=[rank2])<br/>源Store开始等待旧读排空，但尚不释放
    O-->>A: REMOVE_REPLICA(old capability)<br/>通知所有旧holder；本例为rank2
    A->>A: 删除旧capability，禁止从S17启动新read
    A->>A: 等旧capability已经启动的read epoch结束
    A-->>S0: REVOKE_ACK(old capability,last_read_epoch=E77)
    S0->>S0: 等ACK排空<br/>或lease expiry+clock guard+max transfer time
    S0->>S0: S17/G42进入Retire Queue和grace period
    S0->>S0: grace结束且无在途读<br/>Header→FREE，generation 42→43
    S0-->>O: RETIRE_COMPLETE(S17,G42)
    O->>O: quorum移除旧replica descriptor
~~~

| 图中步骤 | 执行者 | 状态变化 | 大白话 |
|---:|---|---|---|
| 1～2 | Owner、新目标Store | rank3/D44/G7进入RESERVED，尚不可读 | 先在目标DRAM上找一个空位D44并占住，但此时里面还没有可用的K7 |
| 3～5 | Owner、旧Store、搬运黑盒 | 保护S17/G42并只读复制payload | 暂时锁住旧HBM副本不让它被释放或覆盖，再把K7复制到D44；复制期间rank2仍可安全读取S17 |
| 6～7 | 新目标Store | 校验完成后D44/G7变COMMITTED并回报Owner | 新位置把数据检查完整后盖上“写完了”的标记，再告诉Owner |
| 8～11 | Owner、rank2 Agent | 发布ADD，安装新capability并返回`ADD_ACK` | 先让目录和rank2知道新DRAM地址；Owner收到确认后，才知道rank2已经具备安全退路 |
| 12～15 | Owner、源Store、rank2 Agent | 旧副本转为RETIRING，并通知所有旧holder撤销旧capability | 现在才通知rank2不要再从S17启动新读；源Store只是进入退役等待，还没有释放S17 |
| 16～17 | rank2 Agent | 等旧read epoch结束后发送`REVOKE_ACK` | rank2让通知前已经开始的读取正常完成，然后回复“我已经不用旧S17了” |
| 18～22 | rank0 Store、Owner | 等ACK或lease安全窗口，经过grace后Free并移除旧descriptor | 确认任何旧读都不可能再访问S17后，才真正释放旧HBM并从目录彻底删除旧地址 |

不可颠倒的顺序是：

~~~text
新slot rank3/D44/G7 Reserve
→ 从旧S17只读复制payload；旧S17继续受保护
→ 新slot COMMITTED
→ Owner发布ADD
→ Decode Agent安装新capability并ACK
→ Owner通知所有旧holder撤销旧副本
→ 等已经开始的旧read排空
→ 旧slot经过grace后才能FREE
~~~

任何一步失败，都可以保留旧 S17 继续服务，或返回 miss；不能产生“目录已经指向新位置，但新 payload 尚未写完整”的窗口，也不能产生“rank2仍可使用旧cap-A，但S17已经被释放或覆盖”的窗口。

#### 2.5.2 主流程的最后一步：旧S17怎样安全交给K20

这一小节**不是另一次 Offload**，只是把 2.5.1 最后的“排空旧读并释放 S17”放大来看。

前面的 Offload 已经解决了 K7 去哪里的问题：K7 仍保存在新的 DRAM 副本中。现在只剩一个本地内存管理问题：

> rank0 原来存放 K7 的 HBM slot S17，什么时候才能重新写入另一个 block K20？

开始和结束状态是：

~~~text
开始：rank0/HBM/S17/G42 = K7 V3
      rank2可能仍持有cap-A，甚至有一次K7读取正在进行

结束：rank0/HBM/S17/G43 = K20 V1
      旧的K7/S17/G42地址已经失效
~~~

虽然物理位置仍叫 S17，但 generation 从 42 变成 43，表示这个 slot 已经进入新的使用周期。可以把它理解成：S17 是房间号，generation 是本轮住户的身份证；房间号没变，但里面已经不是同一个对象。

**本图目标：** 等 rank2 不再使用 `K7/S17/G42` 后，rank0 才把 S17 释放并将 generation 增加到 43，最后把 K20 完整写入 `S17/G43`。下面的步骤就是 2.5.1 中撤销和回收阶段的局部放大，不会再执行一次 K7 数据迁移。

~~~mermaid
sequenceDiagram
    autonumber
    participant A as rank2：旧S17的读者
    participant S0 as rank0 Local Pool Manager
    participant H0 as rank0 HBM slot S17

    Note over A,H0: 初始：S17/G42保存K7 V3；rank2可能仍持有cap-A
    S0-->>A: REVOKE(cap-A)<br/>通知：以后不要再从S17启动新的K7读取
    A->>A: 删除cap-A，禁止新read
    A->>A: 等通知前已经启动的read epoch结束
    A-->>S0: REVOKE_ACK<br/>我已经不再使用K7/S17/G42
    S0->>S0: 确认无有效capability、无在途搬运
    S0->>S0: 等grace period
    S0->>H0: 旧K7 Header→FREE
    S0->>S0: S17 generation 42→43
    S0->>H0: Reserve S17/G43 for K20
    S0->>H0: 写K20 Header(state=WRITING)+payload
    S0->>H0: 校验完成，K20 Header→COMMITTED
~~~

| 图中步骤 | 专业动作 | 大白话 |
|---:|---|---|
| 1～4 | `Revoke → Drain → ACK` | 先让rank2停止新读，再等已经开始的旧读全部结束；此时还不能覆盖S17 |
| 5～6 | 检查capability、在途搬运和grace period | rank0再确认确实没人可能碰旧数据，并额外等待一个安全窗口 |
| 7～8 | `FREE + generation++` | K7正式离开S17；S17从旧身份G42切换成新身份G43 |
| 9～11 | `RESERVED → WRITING → COMMITTED` | 现在才允许把K20写进同一物理位置，并且写完整后才可对外发布 |

最需要记住的是：

~~~text
不是：S17里写入K20 → 再通知rank2旧地址失效

而是：通知rank2停止新读
   → 等旧读结束
   → S17/G42变为FREE
   → generation变为G43
   → 最后写入K20
~~~

正常协议已经通过“先撤销、再排空、最后复用”避免旧读和新写重叠。generation 和 BlockHeader 是最终防线：即使某个异常读者仍拿着旧 descriptor，它期望的是 `K7/V3/S17/G42`，实际看到的是 `K20/V1/S17/G43`，身份校验失败，不能进入 READY。

### 2.6 通知延迟或丢失时，Get 为什么仍安全

假设 rank2 没收到 `REMOVE_REPLICA(cap-A)`：

1. cap-A 有效期间，rank0 必须继续保护 S17/G42；
2. cap-A 剩余时间不足 `max_transfer_time + clock_guard` 时，rank2 不得启动新读取；
3. rank0 只在 lease 到期并经过 guard、最大搬运时间和 grace period 后回收；
4. rank2 到期后会本地拒绝 cap-A，并选择仍有效的 cap-B；
5. 即使实现异常访问了旧 GVA，Header 的 key/version/generation/incarnation 不匹配也不能进入 READY。

因此：

~~~text
通知负责“尽快切换”
lease/read epoch负责“旧读期间不能复用”
Header负责“最终身份校验”
~~~

### 2.7 Remove：逻辑删除和物理回收不是同一时刻

本节是统一环境中的另一条操作分支，用于单独验证删除语义；第 2.8 节的总时间线选择“保留 rank1 DRAM 副本并继续读取”的 offload 分支，不把 Remove 和后续 Get 同时放在一条时间线上。

#### 2.7.1 开始前状态

~~~text
K7 V3仍为READABLE
rank2可能仍持有尚未到期的cap-B
Remove调用者要求删除expected_version=3
~~~

#### 2.7.2 详细时序图

**本图目标：** 从整个 Pool 中逻辑删除 K7 V3，而不是把它搬到其他地方。Owner 先把 K7 V3 标记为 `TOMBSTONE`，使所有新查询返回 miss；随后通知正在使用它的 rank2，以及保存 payload 的 `rank0/HBM/S17/G42` 和 `rank1/DRAM/D81/G9`。两个物理 slot 要等旧读取排空后才真正释放。

~~~mermaid
sequenceDiagram
    autonumber
    participant C as Remove调用方
    participant O as shard27 leader
    participant R as shard27 followers
    participant A as rank2 Decode Agent R1
    participant S0 as rank0 Local Pool Manager
    participant S1 as rank1 Local Pool Manager

    C->>O: RPC Remove(K7,expected_version=3,mode=snapshot)
    O->>O: 校验当前对象仍为K7 V3
    O->>R: 复制TOMBSTONE(seq106)，停止新capability
    R-->>O: quorum确认
    O-->>A: INVALIDATE_OBJECT(K7,V3,seq106)
    O-->>S0: RETIRE_OBJECT(K7,V3,S17,G42)
    O-->>S1: RETIRE_OBJECT(K7,V3,D81,G9)
    A->>A: 禁止新Resolve/新快照；标记旧capabilities撤销中
    A-->>O: ACK(seq106)
    S0->>S0: 等capability/read epoch/grace后Free
    S1->>S1: 等capability/read epoch/grace后Free
    O-->>C: 默认模式返回：逻辑删除已提交
~~~

#### 2.7.3 每一步发生了什么

| 图中步骤 | 状态变化 | 大白话 | 对新读者 | 对已获准旧读者 |
|---:|---|---|---|---|
| 1～2 | 校验`expected_version` | 先确认调用者要删除的确实是当前K7 V3，避免误删后来产生的新版本 | 尚未变化 | 正常读取 |
| 3～4 | TOMBSTONE获多数派 | 多数目录副本记下“K7 V3逻辑上已经删除”，从这一刻起不能再给新读者发许可证 | 新Resolve必须miss | 已授予capability仍按快照语义处理 |
| 5～7 | 定向通知Agent和两个Store | 只通知正在使用K7的rank2，以及实际保存两个副本的rank0/rank1：停止继续使用并准备回收 | 不再产生新capability | 开始撤销/排空 |
| 8～9 | Agent安装失效序列并ACK | rank2先在本地封住K7的新读取，然后回复Owner“删除通知已经收到” | 本地立即拒绝新快照 | revoke前合法读可以完成 |
| 10～11 | 两个Store延迟物理Free | rank0/rank1继续保护旧slot，等已经开始的读取全部结束后才真正释放空间 | 无物理地址可新发布 | 保护旧的在途搬运直到安全结束 |
| 12 | 默认Remove返回 | Owner告诉调用者“从目录看K7已经删掉”；后台物理空间可能还在安全排空 | 逻辑删除完成 | 不承诺所有旧lease已物理排空 |

若业务要求“Remove 返回后旧 capability 也不能再启动读取”，则使用强制 drain 模式：等待所有 holder ACK，或等待 lease 到期及安全时间后再返回。它更强，但 Remove 时延更高。

#### 2.7.4 结束后状态

~~~text
逻辑状态：K7 V3 = TOMBSTONE，新Resolve必须miss
物理状态：S17/G42、D81/G9可能仍在等待旧读排空
最终状态：两个slot分别由本地allocator安全Free
~~~

### 2.8 主线回顾：K7从首次读取、Offload到再次读取

这一节**不引入新协议**，只是把前面已经讲过的操作串成一个容易顺序阅读的故事。

它也不是把第 2 章所有分支硬塞进一张图。下面的时间线只选择一条能够真实连续发生的正常主线：

| 是否进入本图 | 第2章内容 | 原因 |
|---|---|---|
| 是 | 2.1 地址解析与授权 | rank2第一次使用K7时，需要知道副本地址并取得cap-A/cap-B |
| 是 | 2.2 Get和2.3副本选择 | rank2先选择rank0 HBM副本读取K7 |
| 是 | 2.5 Offload及旧slot复用 | rank0释放HBM，rank2切换到已有的rank1 DRAM副本 |
| 否 | 2.4 Put K10 | 这是另一个key的独立写入例子，不是K7主线的一部分 |
| 否 | 2.6 通知丢失 | 这是Offload消息丢失时的异常分支，正常主线假设通知成功 |
| 否 | 2.7 Remove K7 | Remove后K7必须miss，不能再执行本图最后的“从DRAM读取K7”，因此它是互斥分支 |

一句话概括本图：

~~~text
K7先有两个副本
→ rank2第一次问Owner并缓存两个地址
→ 第一次从rank0 HBM读取
→ rank0要释放HBM，rank2切换到rank1 DRAM
→ 旧S17安全写入K20
→ rank2下次直接从rank1 DRAM读取K7
~~~

#### 2.8.1 五个阶段的总时间线

**本图目标：** 验证 capability 缓存能跟随副本变化安全更新：rank0 的 K7 HBM 副本被撤销并复用以后，rank2 不会继续访问旧 S17，而会使用本地已经保存的 cap-B 读取 `rank1/DRAM/D81/G9`。

本图采用 2.5 的快速路径：rank1/D81 已经有可用的 K7 V3，所以不需要再创建 rank3/D44；只需确认 cap-B 可用，然后撤销 rank0/S17。

~~~mermaid
sequenceDiagram
    autonumber
    participant P as Pool事件输入
    participant A as rank2：当前K7读者
    participant O as K7的Owner Shard
    participant S0 as rank0 HBM：S17
    participant S1 as rank1 DRAM：D81

    Note over P,S1: 阶段一：K7的两个副本准备完成
    P->>S0: K7 V3副本A写入并Commit
    P->>S1: K7 V3副本B写入并Commit
    O->>O: 发布K7 V3 replicas=A+B

    Note over A,O: 阶段二：rank2第一次使用K7，只在这时询问Owner
    A->>O: CapabilityLookup miss后批量解析cap-A/cap-B
    O-->>A: 安装K7本地目录

    Note over P,S0: 阶段三：第一次Get，选择rank0 HBM
    P->>A: 输入selected=[K1,K2,K7]
    A->>S0: 不访问Owner，凭cap-A读取K7
    S0-->>A: K7搬回并Header校验READY

    Note over A,S1: 阶段四：rank0释放HBM，rank2切换到已有DRAM副本
    S0->>O: 请求撤销HBM副本A
    O->>S1: 确认DRAM副本B仍可读
    O-->>A: REMOVE_REPLICA(cap-A)
    A->>A: 本地改为只保留cap-B
    A-->>S0: 旧read epoch结束ACK
    S0->>S0: lease/ACK/grace完成
    S0->>S0: S17 G42→G43，保存K20

    Note over P,S1: 阶段五：再次Get，直接使用本地cap-B
    P->>A: 再次输入selected=[K7]
    A->>S1: 不访问Owner，凭cap-B读取K7
~~~

| 图中步骤 | 阶段 | 对应章节 | 大白话 |
|---:|---|---|---|
| 1～3 | 准备K7双副本 | 本章统一环境 | K7先在rank0 HBM和rank1 DRAM各保存一份；两份完整后，Owner才发布它们 |
| 4～5 | 第一次解析地址 | 2.1 | rank2第一次不知道K7在哪，所以问一次Owner，并把cap-A、cap-B缓存到本地 |
| 6～8 | 第一次Get | 2.2、2.3 | DSA选中K7后，rank2本地比较两个候选，直接从预计更快的rank0 HBM读取 |
| 9～13 | 切换读者 | 2.5.1 | rank0想释放S17；系统确认rank1/D81可用后，通知rank2删除cap-A并改用cap-B |
| 14～15 | 回收并复用S17 | 2.5.2 | 等旧读结束后，rank0才把S17从G42变为G43并写入K20 |
| 16～17 | 第二次Get | 2.2 | rank2本地已经只剩cap-B，因此不问Owner，直接从rank1/D81读取K7 |

#### 2.8.2 第2章到这里分别证明了什么

| 要证明的问题 | 结论 | 主要来源 |
|---|---|---|
| 不经过中心MetaService，rank2如何知道K7在哪 | capability miss时只访问K7的Owner Shard，随后把地址缓存到本地 | 2.1 |
| capability命中的Get是否仍访问Owner | 不访问；本地过滤候选后直接搬运payload | 2.2 |
| 多个副本读谁 | 先排除不安全候选，再选择预计完成时间最短的副本 | 2.3 |
| Put写到一半会不会被读到 | 不会；WRITING没有CommitAck，Owner不会发布READABLE | 2.4 |
| rank0换出K7后rank2如何切换 | 先确保DRAM副本可用，再定向通知rank2删除cap-A | 2.5.1 |
| 旧S17什么时候才能写入K20 | 撤销、旧读排空和grace全部结束后，generation递增才能复用 | 2.5.2 |
| Offload通知丢失怎么办 | lease到期前源slot继续受保护；到期后读者本地拒绝旧capability | 2.6 |
| Remove为什么没有放进主线图 | Remove是互斥分支；执行后K7必须miss，不允许主线最后再次读取 | 2.7 |
| 旧GVA已经指向K20怎么办 | generation、incarnation和BlockHeader不匹配，不能进入READY | 2.5.2、2.6 |

到这里已经可以从具体事件检查协议是否闭环。下面再把流程中出现的组件和规则抽象出来。

---

## 3. 从流程中抽取总体架构

### 3.1 什么叫“完全分布式”

本方案不存在：

~~~text
专门的中心MetaService进程
一台机器保存的全量key表
所有rank共享访问的GlobalAllocator
每次Get必须访问的中心目录
所有rank参与的全局广播或全局锁
~~~

每个 rank 都运行相同 PeerService：

~~~text
PeerService(rank N)
│
├── Membership View
│   └── rank列表、node incarnation、membership epoch
├── Owner Shard Replicas
│   └── 只保存分配给本rank的一部分元数据shards
├── Local Pool Manager
│   └── 只管理本rank HBM/DRAM/SSD slot
└── Decode Agent
    └── 仅缓存本rank活跃请求所需capabilities
~~~

所有 Owner Shard Replicas 的逻辑并集可以回答全局 key 问题，但它们从不物理合并成一张总表。

### 3.2 如何直接找到 Owner

每个 rank 本地持有一个很小的 Membership View：

~~~text
MembershipView {
    epoch = 8,
    ranks = [0,1,2,3],
    incarnation = {0:17,1:11,2:4,3:20}
}
~~~

本地执行：

~~~text
shard_id = Hash(key) mod SHARD_COUNT
owner_group = RendezvousHash(shard_id, MembershipView)
~~~

Membership 只包含节点和 shard 映射，不包含任何 KV key。节点变化是低频控制面事件，不进入每个 Decode step。

### 3.3 总体架构图

**本图目标：** 这不是一次 Put/Get/Offload 操作，而是把统一环境中的组件放回各 rank：每个 rank 的 Decode Agent 负责本地快速读取决策，Local Pool Manager 只管理本 rank 贡献的 HBM/DRAM/SSD slot，Owner Shard 分散在各 rank Host CPU 上维护不同 key 的逻辑目录。图中的实线/虚线用于区分控制信息与 KV payload 的流向。

~~~mermaid
flowchart TB
    IN["输入边界<br/>selected keys或sealed payload"]

    subgraph P2["rank2 PeerService / 当前服务R1"]
        A2["Decode Agent R1<br/>Capability Cache + READY + Selector"]
        S2["Local Pool Manager 2<br/>本地K1/K2和预取slot"]
        O2["部分Owner Shard followers"]
    end

    subgraph P0["rank0 PeerService"]
        S0["Local Pool Manager 0<br/>K7 / HBM S17 G42"]
        O0["shard27 follower"]
    end

    subgraph P1["rank1 PeerService"]
        S1["Local Pool Manager 1<br/>K7 / DRAM D81 G9"]
    end

    subgraph P3["rank3 PeerService"]
        O3["shard27 leader<br/>K7版本、副本集合、订阅者"]
        S3["Local Pool Manager 3"]
    end

    X["数据搬运层（黑盒）"]
    M["所有rank本地持有相同Membership View"]

    IN --> A2
    A2 -->|"本地K1/K2"| S2
    A2 -->|"有效capability"| X
    S0 -->|"候选A"| X
    S1 -->|"候选B"| X
    X -->|"payload到本地slot"| S2

    O3 <-->|"仅变更路径：shard内复制"| O0
    O3 <-->|"仅变更路径：shard内复制"| O2
    O3 -.->|"准备/异步更新"| A2
    S0 -->|"Commit/Retire状态"| O3
    S1 -->|"Commit/Retire状态"| O3
    M -.-> O0
    M -.-> O2
    M -.-> O3
~~~

| 路径 | 包含什么 | 稳态 Get 是否经过 |
|---|---|---:|
| 数据热路径 | Decode Agent、本地Pool、数据搬运黑盒、Header/READY | 是 |
| 元数据准备路径 | Owner Resolve、Store保护、capability安装 | 否，提前/后台 |
| 元数据变更路径 | Owner quorum、Store Commit/Retire、通知 | 否 |

---

## 4. 三个核心组件

### 4.1 Owner Shard：管理逻辑真相

一个 Owner Shard 管理很多 keys，而不是一个 key 创建一个 Raft 组。K7 的记录近似为：

~~~text
ObjectRecord(K7) {
    object_version = 3,
    state = READABLE,
    owner_epoch = 61,
    update_seq = 104,
    replicas = [
        rank0/HBM/S17/G42/inc17/COMMITTED,
        rank1/DRAM/D81/G9/inc11/COMMITTED
    ],
    subscribers = [rank2/R1],
    tombstone = false
}
~~~

Owner Shard 负责：

- 串行化同 key Put、发布、迁移和 Remove；
- 只发布 Store 已确认 Commit 的副本；
- 记录订阅者并推送单调 `update_seq`；
- leader换届时生成新的 fencing epoch。

Owner Shard 不负责：

- 不保存全部 ranks 的 free-list；
- 不替 Store 选择具体 slot；
- 不为每次 Get 现场选副本；
- 不搬运 KV payload。

### 4.2 Local Pool Manager：管理物理真相

每个 rank 只允许自己的 Local Pool Manager 修改本地 allocator：

~~~text
SlotRecord(S17) {
    state = COMMITTED,
    key_hash = Hash(K7),
    object_version = 3,
    generation = 42,
    node_incarnation = 17,
    media = HBM,
    gva = GVA-S17,
    protect_until = T100,
    holders = [rank2/R1]
}
~~~

它负责 Reserve、Write、Commit、Retire、Free、capability保护、在途读取排空、generation递增和重启 incarnation 变化。

### 4.3 Decode Agent：管理请求级快速路径

rank2 的 Decode Agent 只保存 R1 可能使用的本地子集：

~~~text
R1 Capability Cache
├── K1 → rank2/HBM/local/READY
├── K2 → rank2/HBM/local/READY
└── K7 V3
    ├── rank0/HBM/S17/G42/cap-A/expires=T100
    └── rank1/DRAM/D81/G9/cap-B/expires=T100
~~~

它负责本地查表、候选过滤、性能选择、read epoch、Header校验、READY、接收副本更新以及不确定时 miss。它不复制全局 Owner 总表。

---

## 5. 流程中使用的核心数据结构

### 5.1 ReplicaDescriptor

~~~text
ReplicaDescriptor {
    key_hash,
    object_version,
    rank,
    media,
    gva,
    size,
    slot_generation,
    node_incarnation,
    replica_state,
    owner_epoch,
    update_seq
}
~~~

它回答“候选副本在哪里”，但本身不代表现在仍可启动读取。

### 5.2 ReplicaCapability

~~~text
ReplicaCapability {
    descriptor,
    capability_id,
    holder_rank,
    lease_expiry,
    capability_epoch,
    integrity_tag
}
~~~

读取方只有满足以下条件才能启动读取：

~~~text
now + max_transfer_time + clock_uncertainty + safety_margin
    < lease_expiry
~~~

### 5.3 BlockHeader

~~~text
BlockHeader {
    magic,
    key_hash,
    object_version,
    payload_size,
    slot_generation,
    node_incarnation,
    state,              // WRITING / COMMITTED / RETIRING / FREE
    header_checksum,
    payload_checksum    // 可按风险和成本启用
}
~~~

Header 验证地址中的对象身份，但不能替代 lease。禁止在读取期间复用源 slot，才是防止 payload 撕裂的基础。

### 5.4 三种本地表不能混为一张

| 表 | 归属 | 保存什么 |
|---|---|---|
| Owner ObjectRecord | Owner Shard复制组 | key的权威逻辑状态和副本集合 |
| SlotRecord/allocator | Storage rank本地 | 本rank具体slot物理状态 |
| Capability Cache | Decode rank请求级 | 当前请求可直接使用的descriptor子集 |

---

## 6. 七条安全不变量

| 编号 | 不变量 |
|---|---|
| I1 | 已 Commit 的同一 `key+object_version` payload 不原地修改 |
| I2 | Owner 只能发布 Local Pool Manager 已确认 Commit 的副本 |
| I3 | capability、在途读取或 grace period 结束前，物理 slot 不能被复用 |
| I4 | slot 每次复用增加 generation；rank 每次重启增加 incarnation |
| I5 | Decode Agent 只使用未过期、未撤销且版本一致的 capability |
| I6 | 只有 Header 校验成功并本地 READY 的 KV 才能交给 Attention |
| I7 | 状态无法证明时 miss/重算，不猜测、不使用未经验证的地址 |

只要 I1～I7 始终成立，通知延迟或丢失不会把错误 KV 交给 Attention；通知只决定切换和回收速度。

---

## 7. 关键并发场景

### 7.1 Get 与 offload 并发

| 时刻 | 处理 |
|---|---|
| 读取已在 revoke 前合法启动 | 源 slot 被 capability/read epoch 保护，允许完成 |
| revoke 已安装、读取尚未启动 | Agent 不再使用旧 capability，选择备用副本 |
| 通知丢失 | capability 到期后禁止新读，Store延迟回收 |
| 搬运或Header校验失败 | 尝试仍有效 fallback，否则 miss |

### 7.2 Get 与 Put 同一逻辑 key

不可变版本允许大部分并发：

~~~text
Get K7 V3
    与
Put K7 V4到新slot

可以并行。Get继续读取已提交V3；新读者只有在V4发布后才发现V4。
~~~

只有以下情况需要等待：

- Get 明确要求尚未 Commit 的 V4；
- 写者试图复用仍受旧读保护的同一物理 slot；
- 两个 Put 竞争同一 key 的 mutation token。

### 7.3 同 key Put 与 Remove

二者进入同一 Owner Shard 日志：

~~~text
顺序A：Put V4 commit → Remove V4
顺序B：Remove V3 → 新Put必须使用明确的新version/generation策略
~~~

不会出现两个 rank 各自决定最终状态。

### 7.4 两个副本报告不同版本

Replica Selector 不按 Store 自己声称的最大版本选择。只有 Owner 当前已提交 ObjectRecord 中的目标版本属于候选；Store Header 只是再次验证。

### 7.5 SSD 回温

SSD 对象不能直接作为普通远端内存 capability：

~~~text
SSD object
→ Storage rank回温到注册DRAM/HBM buffer
→ Commit临时或正式replica
→ 签发capability
→ 才能进入Get候选集合
~~~

---

## 8. 故障语义

本方案首先覆盖 crash-stop、消息丢失/重复/乱序和网络分区，不覆盖恶意 Byzantine rank。

### 8.1 Owner leader 宕机

~~~text
shard27 leader rank3宕机
→ rank0/rank2多数派选出新leader
→ owner_epoch 61→62
→ 恢复已提交ObjectRecord
→ 未提交mutation abort或按日志重放
~~~

Store 只接受当前 Owner epoch 的新 Reserve/Retire；旧 leader 恢复后持有的 fencing token 已失效。

### 8.2 Storage rank 重启

rank0 重启时 `node_incarnation 17→18`。旧 capability 全部引用 inc17；旧读取失败或 Header incarnation 不匹配。Owner 将 rank0 旧 replicas 标为 LOST，K7 仍可从 rank1 DRAM 读取。

### 8.3 Decode rank 宕机

rank2 不再发送 ACK 时，Store 不无限等待进程；旧 capability 到期后自动失去保护权，holder记录由 lease expiry 回收。

### 8.4 网络分区

| 分区一侧 | 能做什么 |
|---|---|
| 拥有 Owner Shard 多数派 | 可以提交元数据变化 |
| 没有多数派 | 不能 Put/Remove/迁移，防止 split brain |
| 持有未过期 capability | 可以继续读取受保护的不可变副本 |
| capability 过期且不能续约 | miss/重算 |

### 8.5 时钟不确定性

Lease 使用单调时钟和已知最大不确定度 `ε`：

~~~text
reader_valid_until = server_expiry - ε - transfer_guard
store_reclaim_after = server_expiry + ε + max_transfer_time
~~~

如果无法给出可接受时钟边界，则回收必须依赖显式 revoke ACK；失联 holder 会让空间保留更久，但不能冒险提前覆盖。

---

## 9. 为什么不会读到错误 KV

### 9.1 当前有效副本

~~~text
Owner已发布
→ Store已Commit
→ capability有效
→ slot未复用
→ Header匹配
→ READY
~~~

### 9.2 正在撤销但 lease 仍有效

Owner 不再给新读者签发旧 capability；旧 holder 仍在 lease 内，Store禁止复用，因此旧读仍正确，只是不一定来自最新位置。

### 9.3 capability 已过期

Decode Agent 在提交数据读取前检查剩余时间，不满足 guard 直接拒绝。

### 9.4 Put 只写了一半

Header仍为 WRITING，Store不发 CommitAck，Owner不发布，因此正常 Resolve 看不到该副本。

### 9.5 旧地址已复用

正常协议通过延迟回收避免重叠；generation、incarnation 和 Header 是最终身份防线，任何不匹配都不能 READY。

### 9.6 多副本性能选择

性能选择只发生在同一个 Owner 已提交版本、有效 capability 的候选集合中，不会改变逻辑版本。

由此得到：

> 任何进入 Attention 的 KV，要么来自仍受保护的已提交 slot，要么已被 Header/READY 拒绝；错误地址、半成品和复用后的对象不能合法进入 Attention。

---

## 10. 与当前 MemCache 源码的关系

### 10.1 当前标准 Get 的同步瓶颈

当前标准 Get 在开始数据搬运前同步调用 Meta：

- [`MmcClientDefault::Get`](../memcache/src/memcache/csrc/client/mmc_client_default.cpp#L347)
- [`BatchGet` 构造请求并 `SyncCall`](../memcache/src/memcache/csrc/client/mmc_client_default.cpp#L407)

当前路径近似：

~~~text
key
→ MetaService SyncCall
→ blob/GVA/lease
→ MmcBmProxy
→ MemFabric
~~~

如果新的 Owner Shard 仍被每次 Get 同步调用，它就只是分片版 MetaService，无法实现本方案目标。

### 10.2 当前已有的直接 GVA 雏形

当前代码已经支持：

~~~text
BatchGetKeyInfo(..., MMC_QUERY_FLAG_GVA_READ_START)
→ LocalGvaBlobTracker::UpdateFromQuery
→ BatchCopy按GVA读取
→ LocalGvaBlobTracker::FindReadable校验lease
~~~

对应源码：

- [`Query` 把 GVA 查询结果放入本地 tracker](../memcache/src/memcache/csrc/client/mmc_client_default.cpp#L597)
- [`LocalGvaBlobTracker::FindReadable`](../memcache/src/memcache/csrc/client/mmc_client_local_gva_blob_tracker.cpp#L195)
- [`BatchCopyReadPath`](../memcache/src/memcache/csrc/client/mmc_client_default.cpp#L1140)

它证明当前接口已经接受“先获得安全 GVA，再按 GVA 搬运”的方向；但现有 tracker 仍接近一次读取事务，不是 DSA 请求级长期 capability cache，也没有完全分布式 Owner 分片和主动副本更新。

### 10.3 组件重构映射

| 当前 MemCache | 新方案 |
|---|---|
| MetaService `metaContainer` | 按 hash 拆成各 PeerService 的 Owner Shards |
| MetaService `GlobalAllocator` | 每个 Local Pool Manager 的本地 allocator |
| `MmcClientDefault` | 增加 Decode Agent、capability preparation 和 fast Get |
| `LocalGvaBlobTracker` | 扩展为请求/工作集级 ReplicaCapability Cache |
| `READ_START/READ_FINISH` | capability lease + revoke/ACK + expiry |
| `BatchGetKeyInfo` | 按 Owner Shard 分组的 `BatchPrepareCapabilities` |
| `BatchCopy` | 保留为 capability 驱动的数据搬运入口 |
| LocalService | 扩展为对等 PeerService / Local Pool Manager |
| MemFabric | 保持数据面职责，不管理 key 一致性 |

建议的概念接口名称是设计占位，不表示当前代码已经存在：

~~~text
ComputeOwnerShard(key, membership_view)
BatchPrepareCapabilities(request_id, keys, preferred_replica_count, lease_duration)
FastBatchGetByCapability(capabilities, destination_buffers)
ApplyReplicaUpdate(update_seq, add_replicas, remove_replicas)
RevokeCapability(capability_id)
AckRevoke(capability_id, last_read_epoch)
~~~

---

## 11. 时延和资源开销

### 11.1 稳态 Get

| 操作 | 次数 |
|---|---:|
| 本地 selected key → capability table 查找 | 每唯一block一次，可batch/向量化 |
| 本地 replica filter/selector | 每block少量候选 |
| 远程元数据 RPC | 0 |
| Owner共识 | 0 |
| 数据搬运 | 仅远端block |
| Header/READY | 每搬回block一次 |

### 11.2 地址解析、授权和后台续约

额外成本包括：

- 按 Owner Shard 批量 Resolve；
- capability grant 和提前续约；
- 每个活跃请求的紧凑 capability entries；
- 副本变化通知与 ACK watermark；
- 粗粒度容量和性能 telemetry；
- 旧 slot 因 lease/grace 暂时占用的额外空间。

### 11.3 变更路径

| 路径 | 额外成本 | 是否阻塞普通 Get |
|---|---|---:|
| 新 Put | Owner quorum + Store Reserve/Commit | 否 |
| 增加副本 | payload copy + Commit + Owner publish | 否 |
| 撤销副本 | Owner mutation + 定向revoke + 延迟回收 | 否，旧lease内仍可安全读 |
| Remove | tombstone quorum + revoke/drain | 新Resolve立即miss；旧快照按语义排空 |
| Owner failover | shard选举和fencing | 有效capability可继续读；过期后可能miss |

### 11.4 DSA 工作集过大时的元数据问题

若长上下文包含大量 blocks，不能盲目为全部 block 保存笨重的独立对象。可以分层：

~~~text
Level 1：本轮selected keys的精确capabilities
Level 2：最近多轮热工作集
Level 3：按segment/range压缩的请求级目录
~~~

这部分是后续原型必须测量的空间与续约流量权衡。

---

## 12. 可行性原型

### 12.1 阶段一：简单纯 DP 健康集群

固定四 rank 和静态 Membership，实现：

- key → Owner Shard；
- 单 leader Owner，不先做故障切换；
- 每 rank 本地 allocator；
- 不可变 KV、BlockHeader、generation；
- request-level capability cache；
- selected keys本地/远端分类；
- 多副本 filter + selector；
- capability驱动Fast Get；
- offload revoke、lease 和 Retire Queue。

必须跑通：

~~~text
K7双副本
→ rank2选择rank0 HBM
→ rank0撤销副本
→ rank2收到通知改选rank1 DRAM
→ 丢通知时等待lease仍不读错
→ S17安全变为K20/G43
~~~

### 12.2 阶段二：并发和失效

加入 concurrent Put、Get/offload、Get/Remove、capability过期、通知丢失/重复/乱序、ACK丢失、partial Put、rank restart/incarnation 和 SSD 回温；每个事件后检查 I1～I7。

### 12.3 阶段三：高可用和规模

加入 Owner Shard 三副本 Raft、membership重配置、fencing token、批量lease、segment/workset subscription、背压、慢订阅者和大规模 key/rank 性能测试。

### 12.4 通过标准

正确性：

- Attention消费的 block，其 key/version/generation/incarnation 均匹配；
- 半成品、过期 capability 和复用 slot 不能 READY；
- 通知丢失和节点重启不产生错误 KV；
- 无法确认时统一 miss。

性能：

- 稳态 selected keys 到第一个远端读取提交之间没有远程元数据 RPC；
- 本地可用KV可以与远端预取重叠；
- 本地命中和有效 capability 命中不访问 Owner；
- Owner/lease 后台流量可以按 request、shard 和 watermark 批量摊销。

---

## 13. 最终总结

贯穿场景的核心状态变化是：

~~~text
初始：
K7 V3 → rank0/HBM/S17/G42 + rank1/DRAM/D81/G9

地址解析与授权：
rank2在capability miss时向shard27取得cap-A + cap-B并缓存在本地

稳态读取：
selected=[K1,K2,K7]
→ 本地查表
→ K1/K2本地READY
→ K7优先rank0 HBM
→ 不访问Owner

rank0撤销：
Owner提交RETIRING
→ 定向通知rank2
→ rank2本地删除cap-A
→ 后续改选rank1 DRAM

安全回收：
ACK/read epoch或lease到期
+ transfer guard
+ grace period

最终：
K7 V3 → rank1/DRAM/D81/G9
rank0/S17/G42 → generation43 → K20
~~~

完全分布式场景下可以让相关 rank 感知副本覆盖或 offload，并在多个正确副本中选择预计最快者。真正保证安全的不是某一条通知，而是：

~~~text
不可变Commit
+ Owner有序发布
+ capability lease
+ 定向revoke
+ read epoch和延迟回收
+ generation/incarnation/Header校验
+ 不确定时miss
~~~

其中 Owner 负责低频地确定“哪些地址合法”，Decode Agent 缓存这些结果；普通 Get 直接使用本地 capability 和数据路径，不把 Owner Shard 重新变成 Decode 关键路径上的分片版 MetaService。
