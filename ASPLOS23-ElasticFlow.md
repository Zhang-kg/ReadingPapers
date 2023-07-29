# ElasticFlow

## HiPO 实验室分享

分享人：fengtianyu



## 总结

**内容和主题**

**结构和组织**

**文体和语言**

**表达效果**

**适合读者**

**创新与价值**

**文章的亮点和缺点**

**上下文和背景**

**综合评价**

## Abstract

ElasticFlow 是一种用于分布式深度学习的弹性无服务训练平台。

ElasticFlow 提供了一个具有两个不同特性的接口（serverless interface）

- 用户只定义深度神经网络（deep neural network, DNN）和超参数（不涉及使用 GPU 的数量）
- 用户定义 job 的 ddl（不包括占用 GPU 的时间）

与现有的以服务器为中心的平台相比，ElasticFlow 在满足最后期限方面提供了性能保证，同时为深度学习开发人员减轻了繁琐、低级和手动的资源管理。

分布式训练有两个挑战：

- 训练吞吐量和 GPU 数量之间非线性扩展关系
- 比例扩张的效率受到 worker placement（我认为是 GPU 位置）的影响

本文提出最小满足共享（Minimum Satisfactory Share）获得训练任务在 DDL 前完成所需的资源，ElasticFlow 在此基础上控制权限。

本文开发了一个贪婪算法，基于收益递减（diminishing returns）的（方式）分配资源给作业。

本文讲伙伴分配（buddy allocation）给 worker placement 来消除拓扑的影响。

对 128 个 GPU 的集群的评估结果显示，和现有方案相比，ElasticFlow 可以多完成 1.46-7.65 倍任务。

## 1. Introduction

当前 DL 模型的训练遵循服务器为中心（server-centric）的模式，DL 开发者以机器实例（machine instance, 例如，物理机，虚拟机或者容器）的形式请求硬件资源来运行 DL 训练作业。

但是服务器为中心的方法有两个限制：

- 以服务器为中心的模型对于 DL 开发者太低级（low level，我觉得可能是底层的意思）了。他们需要显式请求硬件资源并配置机器来运行作业。此外，他们还需要根据物理资源配置超参数（如根据 GPU 显存配置 batch size），这对于不具备系统方面专业知识的 DL 开发者来说具有挑战性。
- 服务器为中心的模型不能灵活地扩展 DL 作业的资源供应，以保证性能需求（如满足 DDL）。性能保证（performance guarantee）在生产情况（production environment）下特别重要，可能需要随时调整模型是否重新训练，或者一些常规任务（比如，使用每日新闻对 BERT 模型进行微调，每天更新推荐服务）。

虽然一些 DL 训练平台不是服务器为中心的（即 DL 开发者不需要知道系统方面的配置），但是他们并不知道 DL 开发人员的性能需求。

**Our proposal.** 我们提出 ElasticFlow，一个无服务器的分布式 DL 训练平台。和现有的 DL 平台相比，ElasticFlow 的关键区别在于，它为 DL 开发者公开了高层次的 DDL 驱动的无服务器接口（high-level deadline-driven serverless interface）具有两个特性：

- DL 开发者只需要提供需要训练的 DNN 模型和任务需要的超参数，不需要指定需要的 GPU 数量
- DL 开发者需要提供 DDL，不需要指定占用 GPU 时间。

DDL 驱动的无服务器接口将 DL 问题和系统管理问题解耦。这种设计可以让 DL 设计者不用再关注低层次的资源管理任务，并且 ElasticFlow 可以利用弹性缩放，为每个作业动态分配资源，保证每个作业 DDL 前完成。

**Challenges.** DL 作业的特点是保证 DDL 的同时给弹性资源分配带来了两个挑战：

- DL 作业的吞吐量和 GPU 数量不呈线性关系。原因是 worker 的通信开销随着 worker 数量增加而增加
- scaling efficiency 受到 worker placement 的影响。同一台服务器内部的 worker 可以使用高带宽的 NVLink 或者 PCIe 进行通信；跨服务器的 worker 带宽较低。

**Key techniques.** 为了在非线性条件下实现最大化资源利用率，我们提出了 Minimum Satisfactory Share。DL 任务的 scaling curves 是凹形的（concave），意味着增加资源可能导致收益递减。我们使用最小满足条件使得 DL 任务能在 DDL 前完成。ElasticFlow 包含两个模块：

- admission control module：负责判断是否接收到达的任务
- resource allocation module：动态分配资源，确保任务的 DDL。对于每个接收的工作，该模块分配满足其 DDL 的最少资源。对于其余资源，模块使用贪心算法，根据 scaling curves 将资源分配给最有效的作业。

我们证明了算法在 concave 的 scaling curves 下是最优的。我们还描述了 ElasticFlow 的扩展以适应 best-effort 的工作（没有 DDL 的工作）。

我们应用伙伴分配解决拓扑依赖位置（topology-dependent placements），这使得 ElasticFlow 将工作分配与准入控制和资源分配解耦。其中任务可以被迁移（jobs can be migrated），worker 的数量也需要是 2 的幂次，伙伴分配保证消除碎片（eliminate fragmentation），也就是说，一个任务总是可以找到一组在拓扑意义上相邻的 GPU，只要空闲 GPU 的数量不小于 job 所需的数量。

**Contributions.** 我们的贡献如下：

- 我们提出了 ElasticFlow，一个用于分布式 DL 训练的弹性无服务器平台
  - ElasticFlow 提供无需服务器的 DDL 驱动接口帮助 DL 开发者人工管理硬件。
  - 利用弹性扩展机制保证 DL 任务在 DDL 前完成任务
- 非线性扩展曲线下，我们提出了满足作业 DDL 需求的最小 GPU 数量。
- 我们设计了一个 admission control 算法，决定哪些任务可以被我们管理，和一个 elastic resource allocation 算法分配任务的资源，确保最大资源利用率
- 我们实现了一个 ElasticFlow 的系统原型（system prototype）并将其与 PyTorch 集成。在 128 GPU 集群评估显示，ElasticFlow 可以满足 DDL 的作业数量比现有最先进的方案多 1.46-7.65x。

## 2. Background and Motivation

### 2.1 Background

**Serverless computing.** 也称为 Function-as-a-Service (FaaS)——功能即服务。传统的 Infrastructure-as-a-Service (IaaS)——基础设置即服务使用以服务器为中心的模型：云服务商提供服务器、虚拟机或者容器。

> 不仅仅有 IaaS，还有 PaaS (Platform-as-a-Service) 和 SaaS (Software-as-a-Service)

传统方式下，需要用于自己决定硬件资源使用等；但是 serverless 条件下，用户只需要关注自己的部分即可（这里原文说的是使用 functions 来编码自己的负载，并将 functions 提交到平台上，Users need to code only their workloads using functions and submit functions to the serverless platform）。另外，low-code development 对于 DL 任务在 serverless computing 下也很有前景。到目前为止，serverless 平台还没有对 GPU 等加速器有成熟支持，所以下一步自然就是确保在 serverless 计算平台支持 GPU 等加速器，从而支持更广泛的工作负载（workload）。

**Distributed and elastic training.** DNN 的训练比较耗时，所以一般通过分布式训练加速 DNN 训练。数据并行（data parallelism）技术是说每个 worker 维护一个 DNN 模型。在迭代中，每个 worker 独立训练，再交换其他 worker 的梯度，整体更新，再开始新的迭代。每个 worker 的 batch size 称为 local batch size，所有 worker 的 batch  size 之和称为 global batch size。也有其他并行方法如 model parallelism, hybrid parallelism and pipeline parallelism。

很多工作针对于优化单个设备上的训练、多个 worker 之间的交流、分布式训练算法还有弹性训练。所以在分布式训练平台上使用弹性训练是一种可行的方法。不过 ElasticFlow 关注的是调度多个训练任务并应用弹性训练的方法在 serverless 条件下保证任务的性能需求。关注的不是应用 elasticity 加速单个训练任务的速度。

### 2.2 Limitations of Existing Solutions

不考虑 DL 作业的特征会导致性能低下，如 Kubernetes or YARN。最近的 efforts 要么遵循以服务器为中心的模型，要么忽略了 DL 开发者的性能要求，又如下两个限制：

首先，服务器为中心的模型对于 DL 开发者太底层了：DL 开发者需要显式申请硬件资源、配置机器运行他们的作业。并且 DL 训练工作受到 GPU 硬件资源的限制。他们遇到了两个问题：

- system problem：基于 GPU 内存适应 local batch size 和决定 worker 数量，这个将影响 global batch size 和训练吞吐量
- DL problem：选择 DL 的超参数

这两个问题在服务器为中心的平台上是交织在一起的（我也有同感，md），对于不具备系统专业知识的 DL 开发者具有挑战性。

其次，现有方案没有通过弹性分配资源从而保证训练任务在 DDL 前完成。大多数解决方案侧重于优化作业完成时间（Job Completion Time, JTC）。虽然对很多场景都有意义，但是还有另一类场景是，当 DL 开发者对作业截至时间有明确期望时。例如，一些生产环境需要模型及时重训并安装模型，便于常规产品发布。

一些 recent work 尝试考虑截至期限，但是他们还是以服务器为中心的，缺乏灵活扩展资源的方式，不能优化集群资源利用并满足 DDL。

## 3. ELASTICFLOW OVERVIEW

### 3.1 Architecture

DL 开发者通过 ElasticFlow 提供的接口向它提交任务，ElasticFlow 根据 DDL 和集群状态动态分配资源给作业。

**ElasticFlow interface.** DL 开发者使用 serverless function 的方式提交作业。一个训练任务 function 包含如下部分：

- DNN model
- Hyperparameters
- Termination condition：作业完成条件，最大迭代次数、准确性等
- Deadline
- Other training components (datasets, optimizer, etc.).

ElasticFlow 和以往的服务器为中心的平台有两个不同：

- 首先，DL 开发者提交 functions，而系统级别的资源管理以及 local batch size 决定这些事情交给 ElasticFlow
- 其次，DL 开发者只需要指定 DDL。对于 DL 开发者不用时刻关注什么时刻结束任务了；对于平台来说也可以自主安排资源的分配和释放了——从而简化了开发者和平台的交互

![ASPLOS23-ElasticFlow-fig1](./../../TyporaImage/image-20230726141806566.png)

**ElasticFlow architecture.** 图 1 展示了 ElasticFlow 的结构。Admission Control 模块决定每个到来的任务是否 admit 或者 drop。

Monitor 将集群状态给 Admission Control 模块，Admission Control 计算该工作的最小需要资源数量。

Resource Allocation 模块调度 admitted 的任务，从而有效利用资源。当某个调度事件，如新任务到来或者某个任务完成，资源分配模块就会分配任务的资源。该模块还会计算戈丁作业的 local batch size（global batch size / GPU 数量）。

Job placement 模块根据拓扑结构选择 GPU，决定 GPU 之后，该模块将任务发送给 elastic training executor，这是一个可以被任何 elastic DL framework 替换的插入组件（plugged-in component）。

Elastic training executor 保证每台机器正确执行 DL 任务。

**Performance guarantee.** 保证每个被纳入系统的任务的 DDL。

### 3.2 Challenges

![ASPLOS23-ElasticFlow-fig2](./../../TyporaImage/image-20230726143433850.png)

**Non-linear scaling.** 吞吐量不会随着 worker 的数量增加而线性增加。DL 的 scaling 曲线通常是凹形的（concave）。如图 2a，集群使用的 8 个 NVIDIA A100 GPU 和 8 个 NVIDIA Mellanox HDR InfiniBand HCAs，内部使用 200 GB/s 的 InfiniBand 网络连接。

ElasticFlow 将会考虑到非线性的 scaling。

![ASPLOS23-ElasticFlow-fig3](./../../TyporaImage/image-20230726144431991.png)

Deadline-aware scheduling 不是一个新鲜的话题：传统使用 Earliest-Deadline-First (EDF) 技术。这种技术将任务按照 DDL 排序，每个任务假设一个 worker，认为整体吞吐量是各自吞吐量之和。这种方法不适合非线性环境。

> 如图 3，假设一个 worker 的吞吐量为 1，两个吞吐量为 1.5；任务 A 和 B DDL 分别是 3 和 3.5；任务 A 和 B job size 都为 3。则此时 EDF 技术就无法完成两个任务（图 3b）；如果每个任务分配一个 worker，就可以完成任务（图 3c）

**Topology-dependent placement.** 
