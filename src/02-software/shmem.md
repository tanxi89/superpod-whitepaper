---
description: "将PGAS与OpenSHMEM置于超节点语境：对称堆全局寻址、单边put/get与原子操作、Scale-Up域Load/Store与Scale-Out RDMA的统一封装，及MoE与KV Cache等稀疏数据流适配。"
keywords: "PGAS,OpenSHMEM,对称堆,put/get,单边通信,AMO,NVSHMEM,Kernel融合,Scale-Up,RDMA"
---

# 共享内存与单边通信

前三节已经说明，统一访存要解决的不是“能不能访问远端显存”，而是“能否以可接受的一致性代价、地址复杂度和调度开销，把跨设备内存稳定兑现为 `Goodput`”。但当远端资源在语义上变得“可访问”之后，新的问题随之出现：这些语义能力如何更直接地暴露到运行时与编程模型中？

如果说集合通信负责组织**规整高吞吐数据流**，那么共享内存与单边通信面对的则更多是**不规整、细粒度、强依赖链**的数据流。对软件层来说，这不是简单的接口选择问题，而是在决定：哪些数据流适合继续留在 collective 语义里，哪些应该下沉到 `put/get/atomic/fence/signal` 这类更贴近内存访问本身的原语里。

对开放超节点体系而言，这里还隐含着另一层要求：若不同硬件平台分别暴露各自的共享内存接口、错误语义、同步方式和调优路径，那么即便底层链路协议已经开放，框架与运维体系仍会在运行时这一层重新碎片化。共享内存与单边通信因此不仅是性能层，也是统一抽象层的重要组成部分。

## PGAS 内存模型与单边通信

与集合通信相比，`PGAS/SHMEM/` 单边通信更接近统一内存语义本身。它们不是脱离前文另起一套模型，而是把 `put/get/atomic/fence/signal` 这类更贴近内存访问的语义，直接暴露为运行时与编程模型接口。

### 2.3.1 超节点中的 PGAS & SHMEM

分区全局地址空间（`Partitioned Global Address Space`）是一类并行编程与运行时模型：把分布式系统中的内存视为“全局可寻址空间”，但同时承认每个处理单元对其本地分区具有更低延迟/更高带宽。其中，全局意味着从任何一个 `PE` 的角度来看，它都可以访问这个地址空间中的任何位置，就好像在访问一个巨大的、统一的内存池一样。分区意味着这个全局地址空间在物理上是分散的，每个 `PE` “拥有”并管理其中的一部分（即它自己的对称堆）。`OpenSHMEM` 是 `PGAS` 路线中最常用的标准之一，其核心概念是**对称对象（symmetric objects）**与**对称堆（symmetric heap）**，并围绕 `put/get`、原子操作以及同步/完成语义提供 `API` 规范。

超节点通信之所以会涉及 `PGAS/SHMEM/` 单边通信，原因不是“它能替代 collectives”，而是：在超节点里存在大量**非规整、细粒度、强依赖链**的数据流（典型如 `MoE token routing`、`KV Cache` 分发/聚合、稀疏更新、异步流水），这类模式用 `AllReduce/AllToAll` 等标准 collectives 表达并不总是自然且高效；相反，“发起方单边 `put/get +` 显式信号同步”的方式能把控制面简化，并更容易进入 `GPU kernel` 融合。

在超节点场景中，引入 `PGAS/SHMEM/` 单边通信的动机通常不止“写起来像共享内存”，而是三点系统性需求：

1. **更细粒度、可组合的通信原语**：集合通信擅长固定模式，但当应用要跨阶段/跨并行组组织数据流（例如推理中的 `KV Cache` 迁移、`MoE` 动态分派、跨层流水）时，单边 `put/get + signal/AMO` 往往更易把通信嵌入算子和调度器[^66]。
2. **设备端发起（kernel 内）与异步任务化**：超节点追求把通信推进从 `CPU` 控制路径中剥离，使其像“设备侧异步任务”一样与计算并行。`Triton-distributed` 明确引入 `symmetric memory`、`signal exchange`、`async-task` 作为核心概念，并展示在每个 rank 同时并行运行 inter-node `P2P`、intra-node `P2P` 与 compute[^67][^68][^69]。
3. **资源池化（尤其推理 `KV Cache`）**：超节点推理系统常出现“prefill 与 decode 分离、`KV Cache` 迁移、历史上下文复用”等需求。`CloudMatrix384` 论文给出一条清晰的数据路径：prefill 完成后通过 `RDMA` 平面把完整 `KV Cache` 迁移到 decode 节点，并把 `KV Cache` 转移与 decode 平面通信隔离；同时，“Context Caching”通过一个分布式存储库组织 paged `KV blocks`，供所有 `NPUs` 访问与复用，并给出 `UB` 平面可将 prefill 吞吐提升至 `1.52x`、`TTFT` 随复用率显著下降等数据点[^67]。这些机制在抽象上与“全局地址 / 全局内存池”的 `PGAS` 目标高度一致。

![对称堆 / registered buffer 示意](imgs/image12.png)
/// caption
`NVSHMEM` 风格的 symmetric heap 与 registered buffer 关系
///

![Attention node 与 Expert node 间的 M2N / N2M 数据流](imgs/image13.png)
/// caption
`MoE / KV Cache` 场景中的跨节点数据流组织
///

### 2.3.2 对称内存与全局地址

对称堆 / 对称对象要求：各处理单元以一致方式分配一段“对称存在”的内存区域，使得同一对象在各 `PE` 上的地址计算具有一致性，从而支持“远端可寻址”的 `put/get` 等单边操作。对称堆管理及相关约束在 `OpenSHMEM` 规范中有明确定义[^5]。在设备侧发起、融合算子与细粒度通信中，“对象定位”不能完全依赖 host 控制面反复分发元数据；对称堆提供了一个稳定的、可缓存的对象寻址基础，使 `kernel` 内的通信可以更少依赖 `CPU` 协调。这也是 `NVSHMEM`、`rocSHMEM`、`Intel SHMEM` 都围绕“对称堆在加速器内存上”组织 `API` 的原因之一[^66]。

下图给出了 `SHMEM` 的内存模型与构建过程。其核心链路可以概括为三步：

1. `UVA`：runtime 提供统一虚拟地址供上层使用，`SHMEM` 通过 `API` 预留 `VA`。
2. `UVA -> PGAS`：`SHMEM` 将预留 `VA` 绑定本卡 / 远端卡物理内存，构造对称堆。
3. `PGAS Usage`：`SHMEM` 保存 `base[PE]`，通过 `malloc` 获取对称堆内空间，再通过对称地址做 `put/get`。

![UVA 到 PGAS 的地址构建示意](imgs/image15.png)
/// caption
`UVA`、各 `PE` 物理地址空间与 `PGAS` 之间的映射关系
///

需要注意的是，`UVA` 能够支撑 `SHMEM` 更好地构建（简化地址管理和控制面交互），但并非一定需要 `UVA` 才能构建 `SHMEM`。此外，构建 `Host+device` 的 `UVA` 同样能更好支持统一寻址与资源池化，但 `SHMEM` 也可以通过自行管理对称堆地址来实现类似功能。而对称内存模型的引入，使得算子的开发变得更加易用与高效。下面是一个最小单边通信示例：

```cpp
#include <stdio.h>
#include <shmem.h>

int main(void) {
    shmem_init();
    int me = shmem_my_pe();
    int npes = shmem_n_pes();

    int *target = (int *) shmem_malloc(sizeof(int));
    *target = -1;

    int my_value = me;
    int right = (me + 1) % npes;

    // one-sided put：右邻居无需配对 recv
    shmem_int_p(target, my_value, right);

    shmem_barrier_all();
    printf("PE %d: target = %d\n", me, *target);

    shmem_free(target);
    shmem_finalize();
    return 0;
}
```

简单来说，`SHMEM` 对称堆提供的是“全局对象寻址的约束”，收益主要体现在：减少地址交换与元数据同步（尤其在 kernel-initiated 模式下）；使 `PGAS` 原语可直接映射到硬件能力（`GPUDirect RDMA`、`GPU-fabric` 原语等），降低控制面复杂度；为 `UCC` 这类 “one-sided collectives” 映射提供基础[^9]。对称堆不是性能优化本身，而是“让单边通信与设备侧融合可工程化”的前提。

### 2.3.3 单边访存与原子操作

`put/get` 将数据移动变成“发起方单边操作”：发起方把本地数据写入远端对称对象（`put`）或从远端读取（`get`），不要求远端同步参与一次配对发送 / 接收；原子操作则提供跨 `PE` 的计数、标志与轻量同步。`NVSHMEM` 官方文档将 signaling operations 定义为“更新远端 flag，并与 `wait/test` 配合提供高效点对点同步”[^79]；`AMO` 则分为 fetching 与 non-fetching 两类，并指出 non-fetching 原子可通过 `quiet/barrier` 强制完成[^75]。

普通的集合通信适合规整同步的数据流；但超节点训练 / 推理中存在大量“非规整数据面 + 规整计算核”的组合：例如 `MoE` 的 token 路由可以看成“按专家 sparse scatter/gather”；`KV Cache` 的搬运也常是点对点 / 一对多的分发。对这类模式，`put/get` 作为数据面，一方面能降低“收发双方协商”的控制复杂度（特别是当发起方已知目标远端对象位置时），另一方面更容易与 `signal/wait` 组合成轻量同步，从而在 `kernel` 内形成可融合的流水[^69]。`Triton-distributed` 将 `OpenSHMEM` 原语集成到编译器与 `Python` 侧，并用信号构造异步任务；其总体加速范围可达 `1.09x-44.97x`（相对 `PyTorch+NCCL/RCCL`），其中包含推理低时延 `AllToAll dispatch/combine` 等通信算子[^69][^51]。

在实现上，`SHMEM` 往往会同时支持 `Scale-Up` 域和 `Scale-Out` 域的单边访存与原子操作。以 `C2C` 和 `RDMA` 为例，两者的差异在于：`Scale-Up` 域内部实现一般基于 `GPU` 的 `Load/Store` 语义（如 `NVLink` 域内数据搬运），而 `Scale-Out` 域则是 `GPU` 触发的 `RDMA Read/Write`（如 `IBGDA`）。二者统一由 `SHMEM` 封装，从而向上层提供一致的 `put/get/AMO` 能力，只是在作用于不同域时，自适应或指定不同的底层物理链路。

![Scale-Up 域单边通信流程示意](imgs/image17.png)
/// caption
基于本地拓扑信息与 heap 管理的 `Scale-Up` 域 `put/get` 流程
///

![Scale-Out 域单边通信流程示意](imgs/image18.png)
/// caption
基于 `RDMA` 注册信息与 `QP` 的 `Scale-Out` 域 `put/get` 流程
///

下面给出一个 `KV Cache` 场景下的 `M:N` 通信算子，用来说明 `SHMEM` 在编写此类稀疏通信算子时的高性能与高易用性：

```cpp
/*
 * KV Cache M:N Disaggregated Transfer (Prefill -> Decode)
 * 核心机制：
 * 1) 1 个 kernel 内并发对多个 Decode PE 做 PUT
 * 2) GPU 直接发起（零 CPU 参与数据面），控制面不需要交互 KV 地址
 * 3) PUT 完成后用 signal 通知，Decode 侧用 wait 阻塞等待
 */
#include "ptshmem.h"

#define NUM_PREFILL_PES 1
#define NUM_DECODE_PES 3
#define KV_SIZE (/* bytes per decode */)

__global__ void prefill_m2n_kernel(const uint8_t *local_kv,
                                   void *kv_sym,
                                   uint32_t *flags,
                                   int num_decode) {
    const int lane = (int)blockIdx.x;
    if (lane >= num_decode) return;

    const int target_pe = NUM_PREFILL_PES + lane;
    const uint8_t *src = local_kv + (size_t)lane * (size_t)KV_SIZE;

    ptshmem_putmem_block_d(target_pe, kv_sym, src, KV_SIZE);

    if (threadIdx.x == 0) {
        ptshmem_signal_u32_d(target_pe, &flags[lane], 1);
    }
}

__global__ void decode_wait_kernel(uint32_t *my_flag) {
    if (threadIdx.x == 0) ptshmem_wait_until_eq_u32_d(my_flag, 1);
    __syncthreads();
    // 就绪后启动 flash_attention7 等计算...
}
```

单边访存与原子操作的收益在于：

1. **细粒度语义**：把“数据移动”与“同步完成”解耦，允许更松散的开始 / 完成同步。将同步从“collective 完成点”细化到“tile/chunk 完成点”能显著提高 overlap 上限，更容易构建 producer-consumer 流水[^82]。
2. **高效率工程**：为 `GPU-initiated` 与融合提供接口点，把“通信启动时机”绑定到 `GPU stream /` 调度器，而不是 host 线程。`GPU-initiated` 通信能减少 `kernel launches`、`CUDA API` 调用与 `CPU-GPU` 同步带来的开销[^33]。

`put/get + AMO` 把复杂通信模式的“控制面”显式化，有利于超节点内部的融合与异步；但需要注意一致性与内存序要求（例如 `fence/quiet` 或等价机制），工程上最容易踩的坑是“弱一致性 + stream 语义 + 进度保证”。`NVSHMEM` 的 `CUDA interactions` 文档给出典型死锁案例：若一个 `PE` 在 barrier 阻塞而另一个 `PE` 等待 signal，且 signal 的 `put-with-signal` 被 barrier 阻塞，则双方无法前进[^84]。

### 2.3.4 通算融合与超级算子

“通算融合”在此特指把通信语义嵌入算子或 `kernel`，以减少 `kernel launch` 边界与全局同步点，实现通信与计算在更细粒度（tile/chunk）上的交错；“超算子（super operator）”则强调跨多个通信原语（甚至跨网络平面 / 层级）组合成端到端数据流的能力（例如 `EP dispatch -> expert GEMM -> combine`，或 `prefill -> KV transfer -> decode`）。

超节点场景下追求的不是“某一个 collective 的峰值带宽”，而是端到端训练 / 推理的迭代时间与尾延迟。当通信比例升高（大模型、`TP/EP` 增多）时，仅靠库内 `ring/tree` 切换往往不足，需要把通信与算子形态协同设计（如把 `ReduceScatter` 或 `AllGather` 的分片与 `GEMM tile` 对齐）[^60]。

这也是业界当前发展的重要方向：

1. `NCCL` 官方文档明确其以“单 kernel”实现 collective，把通信与计算放在同一 `kernel` 中以降低同步与资源开销[^72]。
2. `Triton-distributed` 报告在多类 workload 上相对 `PyTorch+NCCL/RCCL` 加速 `1.09x-44.97x`，并在 `AG+GEMM`、`GEMM+RS` 上给出平均 `1.42x`（相对 `PyTorch+NCCL`）等结果[^69]。
3. `Flux` 提出通过 `kernel fusion` 细粒度隐藏通信，可在融合 `kernel` 中 overlap 最高 `96%` 的通信，并报告在 `128 GPU` 训练上最高 `1.24x` 提升、推理 `prefill/decoding` 上最高 `1.66x / 1.30x`[^86]。
4. `Ascend` 提出 `MoeDistributeDispatch` 和 `MoeDistributeCombine` 两个通算融合算子技术：将计算和传输拆解为 token 粒度的计算单位，通过流水排布实现通信和计算并行执行。

![通信 / 计算 / signal exchange 的异步任务化示意](imgs/image16.png)
/// caption
`symmetric memory` 与 `signal exchange` 组织的异步任务模型
///

![producer GEMM、local reduction 与 p2p kernel 的并行流水](imgs/image19.png)
/// caption
多 `stream` 下的通算重叠执行
///

下面给出一个 `AllGather + GEMM` 融合算子示例，说明 `SHMEM` 在通信计算重叠上的优势：

```cpp
/*
 * AllGather + GEMM 融合算子（一机 4 卡）
 * 核心机制：
 * 1) 单 kernel 融合：通信 blocks 与计算 blocks 在同一 kernel 并发执行
 * 2) 对称堆免交换：gather 缓冲用 ptshmem_malloc，所有 PE 对称
 * 3) 天然重叠：一块数据通信就绪即开始计算
 */
__global__ void fused_allgather_gemm(const float* B, float* C, float* gather,
                                     volatile uint32_t* flags,
                                     int npes, int me, int m_local, int k, int n) {
    int comm_blocks = npes - 1;
    bool is_comm = blockIdx.x < comm_blocks;

    if (is_comm) {
        int peer = blockIdx.x;
        float* my_data = gather + me * m_local * k;
        ptshmem_putmem_block_d(peer, my_data, my_data, m_local * k * sizeof(float));
        if (threadIdx.x == 0) ptshmem_signal_u32_d(peer, &flags[me], 1);
    } else {
        for (int src = 0; src < npes; ++src) {
            if (threadIdx.x == 0) ptshmem_wait_until_eq_u32_d(&flags[src], 1);
            __syncthreads();
            for (int row = blockIdx.x - comm_blocks; row < m_local; row += gridDim.x - comm_blocks) {
                float* Arow = gather + src * m_local * k + row * k;
                for (int j = threadIdx.x; j < n; j += blockDim.x) {
                    float acc = 0.f;
                    for (int kk = 0; kk < k; ++kk) acc += Arow[kk] * B[kk * n + j];
                    C[(src * m_local + row) * n + j] = acc;
                }
            }
        }
    }
}
```

![AllGather + GEMM 融合流水示意](imgs/image20.png)
/// caption
本地数据计算、分阶段 gather 与计算重叠
///

在真正的项目实现层面，需要注意三点：其一，通信分片粒度（`dispatch/combine chunk`）与算子 tile 对齐；其二，执行资源隔离（`Copy Engine / DMA vs SM`）避免互相抢占；其三，同步机制从全局 barrier 下沉到 `signal/wait/AMO`。`Triton-distributed` 的实现明确采用“在 host 侧分配 symmetric memory，并把通信部分与计算部分放到不同 `stream`；在 `GEMM kernel` 内用 `wait/consume_token` 建立依赖”的方式，以达到“计算不等待通信”的效果[^69]。

## SPMD/BSP vs PGAS/单边通信

| 特性 | SPMD + 集合通信（BSP） | PGAS + 单边通信 |
| --- | --- | --- |
| 通信模式 | 双边或集合式，强调阶段性同步 | 单边 `Put/Get/Atomic`，强调异步解耦 |
| 同步边界 | 通常是 communicator 级、阶段级 | 可以缩小到对象级、chunk 级、信号级 |
| 性能风险 | Straggler 放大，屏障拖累长尾 | 需要自行管理一致性与完成语义 |
| 优势场景 | 稠密训练、规整归约、负载较均匀 | `MoE`、长序列推理、`KV Cache` 迁移、异步流水 |
| 软件要求 | 高质量 collective 算法与拓扑调优 | 对称内存、完成通知、设备侧发起与运行时进度机制 |

从系统设计角度看，真正重要的问题不是“选哪一种”，而是**哪些数据流适合留在集合通信里，哪些应该下沉到单边语义里。** 超节点的软件竞争力，很大程度上就体现在这条边界划分是否合理。

## 小结

`PGAS/SHMEM/` 单边通信在超节点中的核心价值不是替代集合通信，而是提供“更细粒度、更可组合、更易嵌入算子与调度器”的通信语义：对称堆 / 全局地址降低远端数据结构的寻址与管理成本；`signal/wait/AMO` 支撑异步任务化与细粒度同步；通算融合与超算子把通信与算子的边界向内收缩，从而提升端到端效率。

未来，`PGAS/SHMEM` 与 `CCL` 的演进边界大致包括三条：

1. 把 one-sided 与 collectives 统一到同一运行时资源模型（如 `UCC` 的 one-sided collectives / 资源抽象）[^9]。
2. 把通信-计算联合优化进一步推向编译器 / `DSL`（如 `Triton-distributed`）。
3. 在 `Scale-Out` 侧引入更强的硬件 offload 与 multicast（如 `SHARP` 及相关方向）。

[^5]: [OpenSHMEM Specification.](https://openshmem.org/specification)
[^9]: [Gorentla Venkata, M. et al. (2021). UCC Overview [Presentation].](https://openucx.github.io/ucc/wg_slides/ucc_am_2021.pdf)
[^33]: [NVIDIA NVSHMEM Developer Guide](https://docs.nvidia.com/nvshmem/archives/nvshmem-101/developer-guide/index.html)
[^51]: [NVIDIA DOCA DPA All-to-all Application Guide. DOCA v2.5.3.](https://docs.nvidia.com/doca/archive/2-5-3/nvidia+doca+dpa+all-to-all+application+guide/index.html)
[^60]: Cowan, M. et al. [MSCCLang: Microsoft Collective Communication Language.](https://parsa.epfl.ch/course-info/cs723/papers/MSCCLang.pdf)
[^66]: [Using NVSHMEM [Documentation]. Version 3.5.21.](https://docs.nvidia.com/nvshmem/api/using.html)
[^67]: Zuo, P. et al. [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/html/2506.12708v2), arXiv:2506.12708v2, 2025.
[^68]: Bachan, J. et al. [Enabling Fast Inference and Resilient Training with NCCL 2.27.](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
[^69]: Zheng, S. et al. [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler](https://arxiv.org/abs/2504.19442), arXiv:2504.19442, 2025.
[^72]: [Overview of NCCL.](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
[^75]: [NVSHMEM Automic Memory Operations [Documentation]. Version 3.5.21.](https://docs.nvidia.com/nvshmem/api/gen/api/amo.html)
[^79]: [NVSHMEM Signaling Operations [Documentation]. Version 3.5.21.](https://docs.nvidia.com/nvshmem/api/gen/api/signal.html)
[^82]: [NVSHMEM Performance Notes.](https://docs.nvidia.com/nvshmem)
[^84]: [NVSHMEM and the CUDA Model.](https://docs.nvidia.com/nvshmem/archives/nvshmem-241/api/docs/cuda-interactions.html)
[^86]: Chang, L. et al. [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858), arXiv:2406.06858, 2024.
