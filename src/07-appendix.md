---
description: "以术语表形式给出全书关键缩写与概念的中英释义，覆盖集合通信、互联与内存语义、并行模式、网络拓扑、RAS及SPI等。"
keywords: "术语表,AllReduce,Scale-Up,Scale-Out,HBD,UCIe,UALink,CXL,OCS,Goodput,SPI"
---

# 附录

## 术语表

| 术语 | 全称 / 说明 |
| --- | --- |
| AllGather | 集合通信原语，每个参与者将本地数据广播给所有其他参与者，最终各方持有完整副本。 |
| AllReduce | 集合通信原语，对所有参与者的数据做归约（如求和）后将结果分发到所有参与者。 |
| AllToAll | 集合通信原语，每个参与者向每个其他参与者发送不同的数据块，MoE 的 EP 并行场景下为主要通信模式。 |
| ATS | Address Translation Service，PCIe/CXL 中由主机侧 IOMMU 提供的地址翻译服务，允许设备使用系统虚拟地址。 |
| CBFC | Credit-Based Flow Control，基于信用的流控机制，接收端按可用缓冲区向发送端发放信用，避免丢包。 |
| CEI | Common Electrical I/O，OIF 发布的通用电气接口规范，定义 SerDes 各代速率与信号完整性要求。 |
| Clos | 一种多级交换网络拓扑（也称 Fat-Tree），通过中间交换层实现无阻塞或近似无阻塞互联。 |
| CPO | Co-Packaged Optics，光引擎与交换芯片共封装，消除可插拔光模块走线，适用于极高密度场景。 |
| CXL | Compute Express Link，基于 PCIe PHY 的缓存一致性互联协议，支持内存池化与异构共享。 |
| DMA | Direct Memory Access，由专用引擎驱动的批量数据搬运，Kernel 分离，适用于大块集合通信。 |
| DP | Data Parallelism，数据并行，每张卡持有完整模型副本、各自处理不同数据批次。 |
| DPU | Data Processing Unit，基础设施平面自治中枢，卸载网络、存储、安全与遥测等非计算负载。 |
| Dragonfly | 一种低直径分层拓扑：节点组成 Group，Group 间通过全局链路互联，兼顾低延迟与高扩展。 |
| EGM | Extended GPU Memory，Grace Hopper 中将远端 LPDDR 映射为 GPU 可直接访问的扩展显存。 |
| EP | Expert Parallelism，专家并行，MoE 模型中不同专家分布在不同设备上，触发 AllToAll 通信。 |
| ESUN | Ethernet Scale-Up Network，基于以太网的 Scale-Up 网络标准。 |
| Flit | Flow Control Unit，链路层最小传输单元；总线型协议多采用固定长度 Flit，以太型采用可变帧。 |
| Goodput | 有效吞吐，扣除气泡、重传、恢复等开销后真正用于有效计算的系统产出。 |
| GMMU | GPU Memory Management Unit，GPU 侧的内存管理单元，负责虚拟→物理地址翻译与访问权限控制。 |
| HBD | High Bandwidth Domain，机柜级高带宽互联域，是超节点的核心系统边界。 |
| HBM | High Bandwidth Memory，高带宽堆叠存储，通过硅中介层（interposer）与计算芯片互联。 |
| IMEX | Import/Export，NVIDIA 跨节点显存共享的用户态守护进程，管理 Fabric Handle 的发布与映射。 |
| KV Cache | Key-Value Cache，Transformer 推理中缓存已计算的 Key/Value 向量，避免重复计算。 |
| LDST | Load/Store，指令级细粒度远端访存，Kernel 融合，适用于 PGAS/SHMEM 编程范式。 |
| LLR | Link-Layer Retry，链路层重传机制，在不依赖上层协议的前提下恢复传输错误。 |
| LPO | Linear-drive Pluggable Optics，线性驱动可插拔光模块，省去 DSP 重定时，降低功耗与延迟。 |
| MoE | Mixture of Experts，混合专家模型架构，通过门控路由激活部分专家子网络，降低计算量。 |
| MTTF | Mean Time To Failure，平均故障间隔时间，衡量系统可靠性的核心指标。 |
| NCCL | NVIDIA Collective Communications Library，NVIDIA 集合通信库，支持多 GPU/多节点集合操作。 |
| NPO | Near-Packaged Optics，光引擎紧邻芯片封装但不共基板，缩短电走线至 <25 mm。 |
| NVLink | NVIDIA 专有高速互联协议，提供 GPU 间低延迟、高带宽的直连通信通道。 |
| NVSwitch | NVIDIA 专用全互联交换芯片，在 Scale-Up 域内实现 GPU 间 any-to-any 通信。 |
| OCS | Optical Circuit Switching，光电路交换，通过光学路径重配置实现拓扑动态调整。 |
| One-Sided | 单边通信，发起方直接执行远端 Put/Get/Atomic，无需对端 CPU 参与或同步配对。 |
| PAM4 | Pulse Amplitude Modulation 4-level，四电平脉冲调制，每符号传 2 bit，是 112G/224G SerDes 的主流编码。 |
| PGAS | Partitioned Global Address Space，分区全局地址空间，每个参与者可直接寻址远端内存分区。 |
| PP | Pipeline Parallelism，流水线并行，模型按层切分到不同设备，依次前向/反向传递。 |
| RAS | Reliability, Availability, Serviceability，可靠性、可用性、可维护性。 |
| ReduceScatter | 集合通信原语，先归约再按参与者切分结果，ZeRO 优化器的核心通信模式。 |
| Scale-Across | 跨园区、跨数据中心的资源协同与联合调度层级。 |
| Scale-Out | 节点或超节点之间的横向扩展，以网络带宽换取更大并行规模。 |
| Scale-Up | 受控域内（通常为机柜级）通过高带宽互联将多个加速器组织为紧耦合系统的能力。 |
| SerDes | Serializer/Deserializer，串行化/解串行化器，物理层基本单元，决定单 Lane 速率上限。 |
| SHARP | Scalable Hierarchical Aggregation and Reduction Protocol，在网计算协议，将归约操作卸载到交换芯片。 |
| SLA | Service Level Agreement，服务等级协议，定义延迟、吞吐等性能约束的量化承诺。 |
| SlimFly | 一种基于 Moore bound 的低直径拓扑，以接近理论最优的最短路径长度实现高带宽密度。 |
| SPI | SuperPod Pareto Index，超节点帕累托指数，本白皮书提出的多维评价与产品注册框架。 |
| SuperNIC | 计算平面的 Scale-Out 出口网卡，提供 RDMA、拥塞控制与多路径等网络加速能力。 |
| TCO | Total Cost of Ownership，总拥有成本，涵盖采购、部署、运维、能耗和折旧的全生命周期费用。 |
| TP | Tensor Parallelism，张量并行，将单个算子的权重矩阵切分到多设备，要求高带宽同步通信。 |
| TPOT | Time Per Output Token，推理场景下每生成一个 token 的延迟，衡量 Decode 阶段性能。 |
| TTFT | Time To First Token，推理场景下从请求到达到首个 token 输出的延迟，衡量 Prefill 阶段性能。 |
| UALink | Unified Accelerator Link，开放总线型 Scale-Up 互联规范，支持 Load/Store/Atomic 内存语义。 |
| UCIe | Universal Chiplet Interconnect Express，芯粒间互联开放标准，定义 D2D 物理层与协议层。 |
| UVA | Unified Virtual Addressing，统一虚拟寻址，使 CPU 与 GPU 共享同一虚拟地址空间。 |
