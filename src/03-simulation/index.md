---
description: 超节点建模仿真方法论：围绕负载建模、系统仿真、校准验证与帕累托分析，量化评估不同超节点方案的系统能力边界、成本与兑现路径。
keywords: 建模仿真,负载模型,系统仿真,帕累托分析,超节点评估,Goodput预测,性能建模
---

# 建模仿真

前两章分别回答了超节点的系统能力边界为何会被打开、以及这些被打开的边界如何通过软件栈被兑现。但对超节点产业而言，还剩下一个更基础的问题：**当不同路线都宣称自己更快、更省、更开放时，我们用什么尺度判断它们处在怎样的系统边界上？**

如果这个问题没有统一方法，后续关于互联、软件栈、参考设计和未来演进的判断，最终都容易退回到规格表比较或经验驱动的局部结论。建模仿真的意义，正是在于把这些离散、局部、易失真的观察组织成一张能够支撑方案判断的边界地图。本章围绕三个问题展开：

1. **为什么是边界比较**——单点 benchmark 和规格表对照为什么不够，多目标帕累托框架为什么更适合超节点决策？
2. **方法链条如何构建**——从负载建模到校准验证，六个环节各自做什么，行业工具基础覆盖到哪里？
3. **边界如何持续移动**——硬件代际、软件迭代和运维变化如何推动前沿外移，为什么建模仿真不能停在一次性结论上？

## 为什么是边界比较 {#pareto-definition}

在多目标优化中，如果方案 A 在所有目标上都不劣于方案 B，且至少在一个目标上严格优于 B，则称 A **支配** B。不被任何其他方案支配的方案即为**帕累托最优解**，所有帕累托最优解在目标空间中的投影构成**帕累托前沿**（Pareto Front）。前沿内侧是当前方案族的可达域，前沿外侧是尚不可达的理想域；边界本身，就是系统能力的上限地图。

对超节点级系统比较，目标向量通常包含但不限于以下维度：

| 目标维度 | 训练侧典型指标 | 推理侧典型指标 |
|:---------|:--------------|:--------------|
| 吞吐 | step time, samples/s | tokens/s/GPU, tokens/s/MW |
| 时延 | 迭代同步延迟, 气泡率 | TTFT, TPOT, P95/P99 |
| 成本 | GPU·hours, $/converged-loss | tokens/10k_rmb, $/1M-tokens |
| 能效 | FLOPS/W, tokens/s/MW | tokens/s/kW |
| 可靠性 | Goodput / 理论峰值, MTTR | SLA 达标率, 可用性 |
| 工程复杂度 | 软件栈适配成本, 运维人效 | 部署周期, 混部管理开销 |

这些维度天然彼此拉扯。更高吞吐往往意味着更复杂的软件组织、更高功耗或更窄的部署条件；更低时延往往要为此付出成本和容量代价。因此，超节点选型的关键不再是"谁有一个最好的点"，而是"谁的边界更靠外、边界在什么条件下成立，以及应当在边界的什么位置取点"。

这一点在推理侧尤为明显。`TP/PP`、`batch`、量化、并行度、`KV` 策略和调度参数会迅速形成 $10^4$ 量级的组合空间；延迟、吞吐、成本、显存、功耗和稳定性之间又天然冲突；突发流量、长尾分布和冷热模型混部则进一步让单点最优难以复用。帕累托分析真正提供的，不是一幅更复杂的图，而是一种"地图视角"：先剔除被支配区域，再分离可行域与不可行域，最后把不同团队的优化主张、线上观测和离线测试纳入同一坐标系比较。NVIDIA 在 Blackwell NVL72 白皮书中披露，仅 72 卡机柜上部署 1.8T MoE 模型，四个并行维度的组合就产生了超过 **2700 种候选配置**——不存在一种方案能在所有指标上同时取胜[^icpe20-pareto-transfer]。

## 方法链条与工具基础 {#methodology}

一旦把问题改写为"判断边界"，方法论也会随之收束。超节点级比较不可能依赖逐点精算——真实候选空间往往在数千甚至上万量级，而高保真推演或端到端实测的单点成本足以让全空间穷举失去可行性。真正可行的路径只能是：**先用足够快的模型把边界大致描出来，再把有限的高成本验证资源压到最关键的区域。**

围绕这一目标，方法链条可以收束为六个连续环节：**负载模型、资源模型、系统模型、校准验证、输出接口、共演进闭环**。负载模型定义需求侧压力，资源模型定义供给侧边界，系统模型把两者压缩成多目标空间中的离散候选点，校准验证在关键区域收紧误差，输出接口把结果沉淀为可持续复用的判断对象，共演进闭环解释边界为何会随负载、软件、硬件和运维共同变化而持续移动。

公开研究和工程实践已经为前三个环节提供了重要基础。训练侧，`ASTRA-sim` 是目前最成体系的分布式训练仿真框架，其核心能力是将计算、通信和显存后端解耦，支持分层网络建模[^astra-sim-1][^astra-sim-2]；`Chakra` 定义了硬件无关、策略中立的负载表示格式[^chakra]。自动并行方面，`Alpa` 等系统的成本模型能在数秒内评估候选配置的计算量、通信量和显存占用[^alpa]。推理侧，`Orca`[^orca]、`Sarathi-Serve`[^sarathi]、`Vidur`[^vidur]、`Splitwise`[^splitwise]、`DistServe`[^distserve] 已覆盖从到达建模、阶段拆分到调度优化的主要环节。

将这些工具映射到六个环节，可以看到哪些环节已有坚实基础、哪些仍是待填空白：

| 方法环节 | 已有工具覆盖 | 覆盖程度 | 待补充方向 |
|:---------|:-----------|:------:|:---------|
| 负载模型 | `Chakra` 执行跟踪规约；`ASTRA-sim` 计算图描述 | ★★★ | 超节点级多租户混合负载画像；推理侧到达模式与 SLA 分布 |
| 资源模型 | `Alpa` 成本模型；`ASTRA-sim` 分层网络后端 | ★★☆ | 国产互联拓扑参数库；能效与可靠性维度扩展 |
| 系统模型 | `ASTRA-sim` 端到端仿真；`Vidur` 推理集群仿真 | ★★☆ | 训练/推理统一的多目标扫描引擎；帕累托前沿自动提取 |
| 校准验证 | 各工具分散的 validation 流程 | ★☆☆ | 统一校准基线与误差置信区间标准 |
| 输出接口 | — | ☆☆☆ | 边界地图的标准化表示与版本管理 |
| 共演进闭环 | — | ☆☆☆ | 负载-软件-硬件联动的边界漂移追踪机制 |

前三个环节已有可直接复用的工业级基础；后三个环节——校准验证、输出接口和共演进闭环——恰恰是把"仿真工具"升级为"持续服务超节点比较的分析能力"所必须补齐的系统性缺口。这里也需要特别强调：**评测不是独立主线，而是服务于校准验证**——真实测量的职责是在关键点上约束模型误差，而不是取代整个建模链条。

当六个环节成立后，建模仿真的结果就会沉淀为两类可持续复用的判断对象。第一类是**当前边界地图**：在统一负载集、统一指标口径和统一校准基线上，不同参考设计位于边界的什么位置、各自适用于怎样的负载画像。第二类是**边界变化假设集**：在已校准的当前边界之上，光互联、`HBM4`、`Chiplet`、`MoE`、长上下文等变量会把前沿向什么方向推、多大程度上缓解现有瓶颈、又会在何处引入新的约束。

## 前沿推移 {#pareto-frontier}

帕累托前沿不是静态资产，而是系统进步是否真实发生的直接度量。所谓"边界外推"，就是整条前沿在多个维度上向更优方向平移，它至少来自两个典型来源。

第一类来源是代际硬件跃迁。以推理场景的 `tokens/s/user vs tokens/s/MW` 坐标系为例，Blackwell NVL72 相比 Hopper NVL8 在甜点配置上实现了约 25–40× 的综合提升，而 Rubin 平台又进一步在吞吐密度和单位 token 成本上将前沿整体外推[^nvidia-rubin-sim]。InferenceX 开源基准在近千块 GPU 上的实测进一步拉大了这一数字：在 DeepSeek R1 大规模 MoE 推理场景下，GB300 NVL72 FP4 disagg+wideEP 相比 H100 FP8 disagg+wideEP 基线，perf/TCO 提升达 9.7×（40 tok/s/user）至 65×（116 tok/s/user）；若跨精度对比（FP4 vs FP8），实测差距甚至达到 100×[^inferencex-v2]。训练侧同样如此：训练 10T `MoE` 模型达到相同目标，Rubin 仅需 Blackwell 约四分之一的 GPU 数量。代际跃迁改变的不是前沿上的取点，而是前沿本身的上界。

第二类来源是同代硬件上的软件迭代。NVIDIA 在 2025 年 8–10 月间的 InferenceMax 基准测试中展示了一个典型现象：在同一 GB200 NVL72 硬件上，仅通过数周的软件迭代，GPT-OSS 推理模型的帕累托前沿就被推出了近 5×[^nextplatform-pareto-sim]。这说明边界不能被视为一次测定后长期有效的稳态对象；如果软件栈、负载画像和运行条件持续变化，模型校准与边界更新就必须成为持续过程。

这种逻辑已经在[方法论验证案例](case-inference.md)中得到具体展开。该案例在 `L40S + A100` 的异构 `PD` 分离架构上，通过 4 种候选架构、3 类请求场景以及多并发、多配置的系统级扫描，描出了帕累托前沿，再结合安全边际校准和成本敏感性分析，区分出哪些收益是稳健的、哪些只是局部改善。下表概括了几类典型优化路径对前沿的影响：

| 优化路径 | 前沿变化特征 | 代表性数据点 |
|:---------|:-----------|:------------|
| MTP（Multi-Token Prediction） | 同硬件上沿吞吐轴外推，时延几乎不变 | tokens/s/GPU 提升 ~30–50%，TPOT 持平 |
| PD 分离（Prefill-Decode Disaggregation） | 打开新前沿分支：时延与吞吐可独立调优 | TTFT P99 降低 2–3×，吞吐密度提升 ~1.5× |
| FP8 量化 | 前沿整体平移：成本维度显著改善 | tokens/s/GPU 提升 ~1.8×，精度损失 <0.5% |
| 规模扩展（Scale-out PD 节点） | 沿前沿延伸但斜率递减：边际收益受网络约束 | 4→16 节点吞吐线性度 ~0.85，>32 节点降至 ~0.7 |

![不同技术路线推移帕累托前沿](imgs/pareto-frontier-shift-by-tech.png)
/// caption
不同技术路线对推理帕累托前沿的外推效应。`MTP`、`PD` 分离、`FP8` 量化和规模扩展不只是把某个点调得更好，而是在不同程度上把整条性能边界往外推。
///

![SLA 约束改变最优点选择](imgs/pareto-sla-constraint.png)
/// caption
`SLA` 约束会直接改变可用前沿。红色虚线代表 `TPOT <= 50 ms` 的约束边界，它提醒我们：理论上更优的点，并不一定是工程上真正可交付的点。
///

从更大的行业视角看，随着超节点从"把更多芯片连在一起"演进为"持续重写系统能力边界"的基础设施形态，建模仿真正在从辅助分析工具上升为一项基础能力：它不仅回答边界究竟在哪里、如何比较，也回答边界将如何继续移动。

[^nvidia-rubin-sim]: Kyle Aubrey. "Inside the NVIDIA Rubin Platform: Six New Chips, One AI Supercomputer." *NVIDIA Technical Blog*, Jan 2026. `https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/`
[^nextplatform-pareto-sim]: Timothy Prickett Morgan. "Software Pushes The AI Pareto Frontier More Than Hardware." *The Next Platform*, Oct 2025. `https://www.nextplatform.com/ai/2025/10/21/software-pushes-the-ai-pareto-frontier-more-than-hardware/`
[^astra-sim-1]: Saeed Rashidi, Srinivas Sridharan, Sudarshan Srinivasan, and Tushar Krishna. "ASTRA-SIM: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms." ISPASS 2020. `https://ieeexplore.ieee.org/document/9238637/`
[^astra-sim-2]: William Won, Taekyung Heo, Saeed Rashidi, Srinivas Sridharan, Sudarshan Srinivasan, and Tushar Krishna. "ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale." ISPASS 2023. `https://arxiv.org/abs/2303.14006`
[^chakra]: MLCommons Chakra Working Group. "Chakra Schema and Tools." 2023-. `https://github.com/mlcommons/chakra`
[^alpa]: Lianmin Zheng, Zhuohan Li, et al. "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning." OSDI 2022. `https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin`
[^orca]: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Sukjoon Lee, and Byung-Gon Chun. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022. `https://www.usenix.org/conference/osdi22/presentation/yu`
[^sarathi]: Amey Agrawal, Nitin Kedia, Ashish Panwar, et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." OSDI 2024. `https://www.usenix.org/conference/osdi24/presentation/agrawal`
[^vidur]: Abhishek Vijaya Kumar, Giorgos Ananthanarayanan, et al. "Vidur: A Large-Scale Simulation Framework For LLM Inference." MLSys 2025. `https://arxiv.org/abs/2405.05465`
[^splitwise]: Pratyush Patel, Esha Choukse, et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." ISCA 2024. `https://arxiv.org/abs/2311.18677`
[^distserve]: Yinmin Zhong, Shengyu Liu, et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI 2024. `https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin`
[^icpe20-pareto-transfer]: Pavel Valov, Jianmei Guo, and Krzysztof Czarnecki. "Transferring Pareto Frontiers across Heterogeneous Hardware Environments." ICPE 2020. `https://research.spec.org/icpe_proceedings/2020/proceedings/p12.pdf`
[^inferencex-v2]: SemiAnalysis, "InferenceX v2: NVIDIA Blackwell Vs AMD vs Hopper", 2026-02. `https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs`
