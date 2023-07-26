# ElasticFlow

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

