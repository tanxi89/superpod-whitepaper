---
description: "论述超节点上集合通信的层级异构与多组并发约束，涵盖拓扑感知算法、AllToAll与MoE场景、拥塞选路与容错机制，及SHARP/NVLS与DPU侧通信卸载。"
keywords: "集合通信,AllReduce,拓扑感知,层级算法,AllToAll,拥塞感知,容错,SHARP,NVLS,SmartNIC"
---

# 集合通信与运行时协同

前三节已经说明，统一访存要解决的不是“能不能访问远端显存”，而是“能否以可接受的一致性代价、地址复杂度和调度开销，把跨设备内存稳定兑现为 `Goodput`”。但当远端资源在语义上变得“可访问”之后，新的问题随之出现：这些语义能力如何被组织成适合真实训练与推理系统使用的通信路径？

本节讨论的正是这一层。**硬件互联再强，如果通信库不能把这些能力组织成稳定、可组合、可调度的通信语义，超节点的带宽和低时延也很难真正兑现为 `Goodput`。** 对软件层来说，集合通信库不是简单地“调用几种 collective”，而是要把内存语义、硬件拓扑、并行策略、拥塞状态和同步关系进一步组织为稳定吞吐。

在超节点中，这个问题比传统集群更尖锐。数据并行、张量并行、流水并行和专家并行会同时存在，而集合通信承担的正是**规整高吞吐数据流**的主路径。对开放超节点体系而言，这里还隐含着另一层要求：若不同硬件平台分别暴露各自的通信原语、错误语义、拓扑接口和调优方式，那么即便底层链路协议已经开放，框架与运维体系仍会在通信库这一层重新碎片化；集合通信库因此不仅是性能层，也是最关键的统一抽象层之一。

## 集合通信语义在超节点上的收益

如果说前三节讨论的是“远端资源如何在语义上变成可访问对象”，那么从这里开始讨论的是：这些对象之间的交互如何进一步被抽象成可复用、可优化的高层通信原语。集合通信因此可以被看作统一内存语义之上的第一层组织形式。

### 2.2.1 超节点中集合通信的特点

超节点通常指在一个逻辑域内，将大量加速卡通过高带宽、低时延互联紧耦合，使其对上层软件呈现“近似单机”的通信特性与资源管理语义。业界已经出现“256/384 级别”的规模化超节点实例：其一，`NVLink Switch System` 通过外置二层 `NVLink` 交换将 `NVLink` 从单机扩展到多节点，可互联最高 `256 GPU`，并在文档中被描述为形成“数据中心级 GPU”[^1]；其二，`Atlas 900 A3 SuperPoD` 可 “pack up to 384 Ascend 910C chips”，并被定位为可协同工作的“单台计算机式”系统[^3]；其三，`CloudMatrix384` 论文披露该超节点跨 `16` 个机柜，包含 `48` 个 `Ascend 910C` 节点（合计 `384 NPU`）与 `4` 个通信机柜（`L2` 交换）[^67]。

在这样的超节点内部，集合通信不再只是“跨节点 `MPI` 的一个实现细节”，而是承载了多种并行策略的关键数据路径：例如数据并行/`ZeRO/FSDP` 典型依赖 `AllReduce`/`ReduceScatter`/`AllGather`；张量并行在层内会频繁触发 `AllReduce`/`AllGather`；`MoE` 专家并行的 token dispatch/combine 往往抽象为 `AllToAll`/`AllToAllv`；这些原语在超节点尺度上会被反复调用并直接决定迭代时间与尾延迟[^13]。

超节点集合通信与“普通单机/小集群集合通信”的核心区别主要体现在三点。

第一，**互联呈层级化且异构**：同一通信域内同时存在多种链路形态与带宽/时延数量级差异（例如 `NVLink/`片内互联 vs `IB/RoCE`；或超节点内 `UB` 平面 vs `RDMA` 平面），这使得“单一算法/单一拓扑”难以在全域最优[^1]。

第二，**通信并发与多租户竞争更加常态化**：训练/推理任务在超节点内会出现多通信组（多个 communicator）并发，且通信模式由并行策略决定（`DP/TP/PP/EP` 混合），导致拥塞与热点不再是“边缘情况”，而是设计前提。阿里云在其数据中心网络论文中指出，通过在集合通信库内部按连接拥塞状态选路，在 `512 GPU` 上四个 `AllReduce` 并发可提升最高 `34.7%`[^15][^69]。

第三，**可靠性成为一阶约束**：规模增大使故障频率与故障影响显著提升。来自大规模研究集群的实测显示，`1024-GPU` 作业 `MTTF` 仅 `7.9` 小时，并随规模可预测下降；甚至预测 `16,384 GPU` 作业 `MTTF` 约 `1.8` 小时[^6]。在 `Scale-Up` 域内，单卡故障会通过并行度结构放大为显著吞吐损失，例如 `TP64` 下 `0.1%` 故障可放大至近 `10% GPU` 吞吐损失[^7]。因此，超节点集合通信必须天然考虑快速检测、隔离、降级与恢复机制，而不仅是“无故障时的峰值带宽”。

基于上述差异，适配超节点集合通信的关键能力可概括为：**分层拓扑感知**、**拥塞避免与可靠容错**、**通信下沉与软硬协同**。

### 2.2.2 超节点拓扑感知与层级算法

“分层拓扑感知”指集合通信库能识别并抽象通信域内的层级结构（卡内/机内/机间/超节点内/跨超节点），并基于层级差异选择或合成合适的逻辑拓扑（`ring/tree/`分层 `ring-tree` 组合等），以在不同消息规模与并行策略下接近最优。相关能力不仅包括拓扑发现，也包括**按层划分通信阶段（phase）**、按消息大小切换协议、以及在多层之间进行分片/聚合（aggregation）策略设计[^13]。因为在超节点内部常见“快域 + 慢域”并存：例如 `NVLink` 域内带宽远高于机间 `RDMA`，若采用全域 `ring`，则慢域链路成为迭代瓶颈；而采用层级算法把跨慢域的数据规模压缩到最小（典型为 `1/K` 的缩减），可显著降低跨域时间占比[^13]。

工业库普遍采用多算法并通过拓扑/消息大小进行切换。`xCCL` 综述在算法模型中给出 `ring AllGather/AllReduce` 的代价公式，并指出 `AllReduce ring` 由 `ReduceScatter + AllGather` 两阶段构成[^13]；同时，该综述也明确列出 `ACCL` 的混合 `AllReduce`：**intra-node `ReduceScatter`（ring） -> inter-node `AllReduce`（halving-doubling） -> intra-node `AllGather`（ring）**[^13]。另一方面，针对“非对称层级/跨超节点卡数不一致”等超节点常见问题，`HCCL` 的开源文档直接指出其提供 `Mesh/Ring/RHD/NHR/NB/AHC/Pipeline` 等多种（含层级与非均衡）算法，并用 `alpha-beta-gamma` 模型评估算法耗时[^26]；其中 `AHC` 文档给出当通信域跨多个层次且层间带宽收敛、甚至分组卡数不一致时的三步流程（分组、组内 `ReduceScatter`、逻辑同号卡组间 `AllReduce`）[^27]。

更进一步，近五年研究趋势是“可编程/可合成”的集合通信：`MSCCLang`（ASPLOS'23）提出 `DSL +` 编译/运行时，将自定义 `AllReduce/AllToAll` 算法编译为单 `CUDA kernel`，相对手工实现的性能提升 `AllReduce` 最高 `1.9x`、`AllToAll` 最高 `1.3x`，且在 `256 A100` 训练与线上服务中获得 `1.10x-1.89x` 的加速[^60]。类似方向还包括 `NVIDIA`[^53]、`Microsoft`[^29] 和 `TACCL`。这类方法在超节点上的价值在于：当物理拓扑复杂、同时存在多原语混叠（例如 `TP+EP`）时，固定的 `ring/tree` 往往无法覆盖“作业特定最优”，而合成/编译式方案能把拓扑、消息大小、端口数、并行度结构一起纳入搜索/生成空间[^60]。

在超节点场景中，层级 `AllReduce` 常见拆分范式之一是：**（层内）`ReduceScatter` -> （层间）`AllReduce` -> （层内）`AllGather`**[^60]。这一拆分可用 `alpha-beta-gamma` 模型进行工程化估算，用于指导切分粒度、并行度与协议选择。工程落地通常包含三步：

1. 拓扑发现与分组：识别 `NVLink/NVSwitch` 域、`PCIe` 亲和、`NIC` 归属与 rail 划分。
2. 按消息大小/并行策略选择算法族：`ring/tree/RHD/`层级 `ring-tree` 等。
3. 跨层数据分片与聚合：确定 `ReduceScatter` 粒度、是否做 chunk pipeline；并在多 communicator 并发时做资源隔离（channel/stream/QP` 配额），避免合成的最优在并发时退化为拥塞[^13]。

层级算法的收益主要体现在“跨慢域的字节数下降”和“跨域同步点减少”。在真实系统中，收益通常随三类因素增大：域内外带宽差距越大（例如 `NVLink >> IB/RoCE`）[^1]；`K` 越大（例如超节点 `Scale-Up` 域从 `8/16` 扩到 `64/256`）[^1]；通信模式越偏“高频小消息”（`TP/`推理解码）或“高并发多 communicator”（多任务/多租户）[^68]。

![层级 AllReduce 拆分示意](imgs/image5.png)
/// caption
intra-node/inter-node 分层 `ReduceScatter` 与 `AllGather` 拆分
///

## All-to-All 性能分析

`AllToAll` 是最能暴露超节点通信问题的原语之一，因为它同时放大了小包抖动、大包拥塞与负载不均。对 `MoE`、分段推理和稀疏专家路由而言，问题往往不在“总带宽够不够”，而在于流量是否集中打向少量专家或少数链路，导致热点持续堆积。

对稀疏 `AllToAll` 而言，通信库至少需要具备三种能力：一是**拓扑感知分组**，知道哪些路径属于同一快域、哪些必须跨域；二是**消息分片与流水推进**，避免大消息独占链路；三是**路由感知或负载感知调度**，在多个可用 rail、多个 `QP` 或多个连接之间主动避开热点。对超节点软件栈来说，这类能力决定了高带宽互联能否真正转化为训练和推理系统里的有效吞吐。

### 2.2.3 超节点拥塞避免与可靠容错

超节点规模上去后，拥塞与故障不再是偶发。拥塞方面，超节点同时承载多条通信链（`DP` 梯度归约、`TP` 层内交换、`EP` token shuffle），且多任务并发会把“理论无阻塞拓扑”变成“现实多流拥塞拓扑”[^15]。故障方面，作业 `MTTF` 随规模快速下降：`1024 GPU` 作业 `MTTF 7.9` 小时；并预测 `16,384 GPU` 作业 `MTTF 1.8` 小时[^6]。对推理/训练系统而言，这意味着“默认重启”会造成显著的 `GPU-hours` 浪费与尾部 `SLO` 违约风险。

这就要求超节点的集合通信拥有两类紧耦合能力：

1. **拥塞感知的路径/rail 调度**：在多 `NIC`、多 rail、甚至多 `QP` 的物理条件下，集合通信库能够对流量进行条带化（striping）、避堵（hotspot avoidance），并进行组间公平（inter-communicator fairness）。
2. **面向故障的持续运行（resilience）**：当链路抖动、`NIC` 故障、超时发生时，通信层能够快速检测并把故障影响从“全局 abort + 重启”缩小为“局部迁移/降级/重构”，对训练 `goodput` 与推理尾延迟提供可解释的上界。

在拥塞调度上，阿里云 `HPN` 论文给出“在集合通信库内部”进行拥塞感知选路的实例：利用连接的 `WQE` 计数器作为拥塞信号，选择最不拥塞连接发送；在 `512 GPU` 上四个 `AllReduce` 任务并发时，性能可提升最高 `34.7%`[^15]。在故障容错上，`R²CCL`（2026）指出在大规模 `ML` 训练/推理中，网络故障可因恢复慢而浪费 `10-15%` 的 `GPU hours`，并提出利用 `multi-NIC` 冗余进行快速连接迁移、带宽感知重分配与弹性集合算法[^43]。其机制包括：`OOB` 通知与探针定位将检测时间从分钟降到毫秒量级、`DMA-buffer rollback` 支持无损重传、以及在 `512 GPU` 仿真中故障数从 `1` 到 `10` 时迭代开销仅从 `1.5%` 增到 `4.3%`（sub-linear）[^43][^13]。此外，训练系统层面的“故障放大”也必须被通信层理解：`NTP` 研究指出在 `TP64` 规模下，`0.1% GPU` 故障可导致近 `10% GPU` 无效产出，并提出在失败发生后把 `TP` 度降低以把吞吐损失压到接近“故障比例”[^7]。它直接说明“超节点集合通信的并行组结构（`TP/DP` 映射）”与可靠性是强耦合问题。

拥塞感知 `multi-rail` 的工程要点可拆为四层：

1. **GPU-NIC 亲和与 rail 绑定**：确保通信线程块/通信 stream 与具体 `NIC/rail` 的映射稳定可控，避免流量“随机落点”导致热点[^67]。
2. **条带化与多 `QP` 并行**：将大消息切分为 chunk，在多 rail 上并发推进，并维持 per-rail completion/credit 机制；该层通常直接决定峰值带宽是否可达[^40]。
3. **拥塞信号与在线选路/调度**：引入低开销拥塞信号（如 `WQE drain rate`），并在通信库内部进行“每次发送的连接选择”[^15]；对多 communicator 并发，还需要“全局公平/优先级”概念，否则单 communicator 的局部最优会转化为系统层面更糟的拥塞[^41]。
4. **故障检测、隔离与无损迁移**：快速检测（`OOB` 通知 + `RDMA probe`）[^43]、无损迁移（`multi-NIC` 注册 + `DMA rollback`）[^43]、算法重构（快速剔除故障边并必要时降级到带宽较低但可持续的拓扑），以及面向框架提供“可恢复错误语义”。

在现有生态中，通信错误处理往往以 “abort communicator” 为主。例如 `PyTorch` 文档指出启用 `NCCL` 异步错误处理后，collective 会被异步 abort 且进程崩溃；blocking wait 可把错误返回给用户但会有性能开销[^44]。这进一步说明超节点需要“通信层 + 系统层”的联合容错方案，而不仅是库级别的错误码。拥塞调度的收益往往以“并发场景提升”体现（如 `34.7%`）[^15]；容错收益则以“避免全局重启与 checkpoint I/O”体现。`R²CCL` 的核心价值在于把训练/推理从“故障 -> 重启 -> 重放/恢复”变成“故障 -> 迁移 -> 继续”的快速路径，从而避免 `10-15% GPU-hours` 的浪费[^43]。

![双 ToR 冗余与故障切换示意](imgs/image8.png)
/// caption
stacked / non-stacked dual-ToR 场景下的数据面与控制面切换
///

### 2.2.4 超节点通信下沉与软硬协同

“通信卸载”指将集合通信的部分数据搬运、规约计算或进度推进从 `CPU` 甚至 `GPU SM` 上移出，交由网络交换芯片、`SmartNIC/DPU`、专用 `DMA/Copy Engine` 或交换侧计算单元执行，以减少端侧介入、降低同步成本并提升规模稳定性。在超节点尺度上，端侧（`CPU/GPU`）承担全部集合通信进度不仅带来额外计算开销，更引入系统噪声和尾延迟抖动；且当并发通信组增多时，端侧线程/核的抖动会放大为全局 barrier 的尾部。硬件卸载的关键价值在于降低“需要全局一致的同步点”对系统噪声的敏感性[^29]。

这里主要涉及两个方向。

其一，**网络内规约（in-network reduction）**。`SHARP` 官方文档说明其通过把集合操作从 `CPU/GPU` 卸载到网络并减少数据在网络中的重复穿越，从而 “dramatically reduces collective operations time”[^49]。在 `Frontera` 全系统规模（`7,861 nodes`）评测中，基于 `SHARP` 的设计使 `AllReduce` 延迟最多降低 `5.1x`（同时 `Reduce 5.4x`、`Barrier 7.1x`）[^29]。`NCCL` 的 release notes 明确列出 “inter-node algorithms for NVLink SHARP: `NVLS`、`NVLSTREE`”[^22]。云上实测也开始公开：`Azure HPC` 博文在单节点上报告 `NVLS` 模式 `16GB AllReduce` 的 `BusBw` 可达 `481.42 GB/s`，并将其归因于 `NVLink SHARP`（在 `NVSwitch` 上卸载算术以减少数据传输）[^50]。

其二，**`SmartNIC/DPU` 卸载**。`DOCA DPA All-to-all guide` 给出 `DPA` 加速 `all-to-all` 的参考实现，并明确“由 `DPA` 线程执行 `all-to-all`，`CPU free`”[^51]。`UCC` 的 `DOCA UROM` 示例同样展示将 `UCC all-to-all` 卸载到 `DPU` 侧 progress queues/threads[^52]。研究上，针对 multicast-based `Broadcast/AllGather` 的工作展示了将 collective progress engine 卸载到 `DPA`：`DPA` 由 `16` 个 `RISC-V` 核、每核 `16` 硬件线程构成，并可在 `DPA kernel` 中发起 `RDMA` 与 `Atomic`[^53]；在其实验中，`DPA` 单线程 `UC` 接收路径吞吐达 `11.9 GiB/s`，且 `128` 线程可支撑等效 `1600 Gbit/s` 的 chunk 处理率（用于推演下一代 `Tb/s` 链路）[^53]。

通信卸载的实现模式常见三类：

1. **网络内汇聚树（aggregation tree）**：把规约算子放到交换侧（如 `SHARP aggregation node`），并通过树形汇聚减少网络中重复转发；优点是延迟/带宽随规模更平滑，缺点是对交换能力与数据类型/算子支持有约束[^48]。
2. **域内交换卸载（`NVLink SHARP/NVLS`）**：把域内规约放到 `NVSwitch` 等交换侧硬件，从而在 `GPU` 间传输的同时完成规约，减少 `GPU` 端搬运与规约次数；该类通常与库的算法选择（`NVLS vs Tree`）共同决定收益与适用消息区间[^22]。
3. **`SmartNIC/DPU` 进度卸载**：把拆包/重组/重传等“低 `IPC` 的数据搬运逻辑”放到 `DPA/DPU` 线程，利用硬件多线程隐藏 `load/store` 延迟，并通过 `DMA engine` 与 `CQ` 事件驱动推进；该类适配面广（不仅限 reduction），但需要处理一致性、缓冲区注册以及与主通信库的协同[^53]。

通信卸载与软硬协同的收益常以“延迟倍数下降 / `CPU` 占用下降 / 尾延迟收敛”呈现：`SHARP` 在全系统规模 `AllReduce` 延迟可降 `5.1x`[^29]；`NVLS` 在单节点 `16GB AllReduce` 上体现了域内卸载的峰值潜力[^50]；`DPA` 卸载则展示了在极高速链路下以少量 `DPA` 线程支撑线速的可能性[^53]。

![传统归约与 NVIDIA SHARP 对比](imgs/image9.png)
/// caption
传统路径与 `SHARP in-network computing` 的对比
///

![BlueField-3 上的 multicast / allgather 卸载示意](imgs/image10.png)
/// caption
`DPU / Datapath Accelerator` 参与集合通信进度推进
///

![SHARP 聚合树示意](imgs/image11.png)
/// caption
物理拓扑与单棵 `SHARP tree` 的映射
///

### 小结

超节点集合通信的收益可概括为三项：**分层拓扑感知**解决“层级不均质导致的慢链路与热点”；**拥塞避免和可靠容错**解决“规模化下的拥塞与尾延迟”以及“系统可用性”；**通信下沉与卸载**解决“`CPU/GPU` 参与通信的结构性开销”[^26]。

其主要局限也同样明确：其一，**自动调优 / 故障恢复仍有工程门槛**，拓扑-协议组合复杂，可能出现性能退化甚至正确性风险[^22]；其二，**算法选择与调度往往是“库内局部最优”**，与系统级多租户调度仍存在 gap；其三，**通信卸载会把“跨原语/跨阶段”的更深度优化推到台前**，这正是共享内存与单边通信要继续承接的语义基座与工程边界问题。

[^1]: [NVIDIA NVLink SGXLS10 Switch Systems User Manual.](https://docs.nvidia.com/networking/display/sgxh100/introduction)
[^3]: Xu, E. (2025) [Groundbreaking SuperPoD Interconnect: Leading a New Paradigm for AI Infrastructure.](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech)
[^6]: Kokolis, A. et al. [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/pdf/2410.21680)
[^7]: Arfeen, D. et al. [Nonuniform-Tensor-Parallelism: Mitigating GPU failure impact for Scaled-up LLM Training](https://www.pdl.cmu.edu/PDL-FTP/BigData/2504.06095v1.pdf), arXiv: 2504.06095v1, 2025.
[^13]: Weingram, A. et al. [xCCL: A Survey of Industry-Led Collective Communication Libraries for Deep Learning.](https://jcst.ict.ac.cn/fileup/1000-9000/PDF/JCST-2023-1-11-2894-166.pdf)
[^15]: Qian, K. et al. [Alibaba HPN: A Data Center Network for Large Language Model Training.](https://regmedia.co.uk/2024/06/27/supplied_alibaba_hpn_paper_2.pdf)
[^22]: [NVIDIA Collective Communication Library (NCCL) Release Notes.](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2181/pdf/NCCL-Release-Notes.pdf)
[^26]: [Ascend/cann-hccl: 华为集合通信库](https://gitee.com/ascend/cann-hccl)
[^27]: [AHC 算法描述](https://gitee.com/ascend/cann-hccl)
[^29]: Microsoft 在集合通信合成或编译式优化方向的相关资料。SHARP 资料。[暂时无法访问](https://par.nsf.gov/servlets/purl/10249393)
[^40]: [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
[^41]: Cao, J. et al. [Crux: GPU-Efficient Communication Scheduling for Deep Learning Training](https://dl.acm.org/doi/10.1145/3651890.3672239)
[^43]: Wang, W. et al. [Reliable and Resilient Collective Communication Library for LLM Training and Serving](https://arxiv.org/pdf/2512.25059), arXiv:2512.25059, 2025.
[^44]: [PyTorch. 分布式通信包 - torch.distributed[EB/OL].](https://pytorch.cadn.net.cn/docs_en/1.12/distributed.html)
[^48]: [NVIDIA SHARP Documentation.](https://docs.nvidia.com/networking/display/sharpv350)
[^49]: [NVIDIA. Scalable Hierarchical Aggregation and Reduction Protocol (SHARP) [White Paper]. Revision 3.12.0.](https://docs.nvidia.com/networking/display/nvidia-scalable-hierarchical-aggregation-and-reduction-protocol-sharp-rev-3-12-0.0.pdf)
[^50]: Affaticati, H. et al. [Performance at Scale: The Role of Interconnects in Azure HPC & AI Infrastructure.](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/performance-at-scale-the-role-of-interconnects-in-azure-hpc--ai-infrastructure/4427238)
[^53]: Khalilov, M. et al. [Network-Offloaded Bandwidth-Optimal Broadcast and Allgather for Distributed AI](https://arxiv.org/html/2408.13356v1), arXiv:2408.13356v1, 2024.
[^60]: Cowan, M. et al. [MSCCLang: Microsoft Collective Communication Language.](https://parsa.epfl.ch/course-info/cs723/papers/MSCCLang.pdf)
[^67]: Zuo, P. et al. [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/html/2506.12708v2), arXiv:2506.12708v2, 2025.
[^68]: Bachan, J. et al. [Enabling Fast Inference and Resilient Training with NCCL 2.27.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
[^69]: Zheng, S. et al. [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler](https://arxiv.org/abs/2504.19442), arXiv:2504.19442, 2025.
