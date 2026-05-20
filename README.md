# Mini-SGLang 学习手册

> 按顺序读，每章末尾都有自查清单（带折叠答案）。
>
> 这套文档目标：让你把 mini-sglang 的代码全跑通，理解它每个设计决策背后的取舍。读完后能自如回答"vLLM/SGLang 的 prefix cache 怎么实现"、"CUDA Graph 在 LLM 推理里有哪些坑"、"为什么要拆多进程"这类系统题。

## 章节

| # | 文件 | 主题 | 你会学到 |
|---|------|------|---------|
| 0 | [推理引擎要解决什么问题](./00-推理引擎要解决什么问题.md) | 动机 | 三大矛盾 + mini-sglang 额外面对的两个矛盾（多进程、CPU/GPU 协同） |
| 1 | [多进程流水线总览](./01-多进程流水线总览.md) | 架构 | API server / Tokenizer / Scheduler / Engine 的拓扑、ZMQ vs NCCL 分工 |
| 2 | [一次请求的完整旅程](./02-一次请求的完整旅程.md) | 调用链 | HTTP → token 流的 6 跳；abort / 流式 / SSE 的实现细节 |
| 3 | [核心数据结构 Req / Batch / Context](./03-核心数据结构-Req-Batch-Context.md) | 数据结构 | cached_len / device_len / extend_len 的语义；为什么 Context 是全局变量 |
| 4 | [KV Cache 池与 Page 分配](./04-KVCache池与Page分配.md) | 内存模型 | MHAKVCache 6 维张量布局；token-level page_table；dummy page 的来历 |
| 5 | [Radix Cache 与前缀复用](./05-RadixCache与前缀复用.md) | 前缀缓存 | RadixTreeNode 的 split_at；引用计数；evict LRU；CacheManager.cache_req 的 4 状态 |
| 6 | [Scheduler 与四个 Manager](./06-Scheduler与四个Manager.md) | 调度核心 | Prefill/Decode/Cache/Table Manager 的协作；token budget；prefill-first 策略 |
| 7 | [Overlap Scheduling](./07-Overlap调度.md) | 性能特性 | overlap_loop vs normal_loop；两个 CUDA stream 协同；finished_reqs 防双 free |
| 8 | [Engine 与 forward_batch](./08-Engine与forward_batch.md) | Engine 内核 | meta tensor + 流式加载；Sampler；ForwardOutput 三字段；shutdown 顺序 |
| 9 | [Attention 后端抽象与三种实现](./09-Attention后端抽象与三种实现.md) | Attention | FA/FlashInfer/TRT-LLM 差异；HybridBackend；为什么 metadata 必须独立计算 |
| 10 | [CUDA Graph 与 Chunked Prefill](./10-CUDAGraph与ChunkedPrefill.md) | 优化 | CUDA Graph capture/replay；pad_batch 桶化；ChunkedReq 状态在 PendingReq 间流转 |
| 11 | [Tensor Parallelism 与 PyNCCL](./11-TensorParallelism与PyNCCL.md) | 分布式 | TP 数学；5 种 Linear；为什么自己写 pynccl；NCCL bootstrap |
| 12 | [MoE 实现](./12-MoE实现.md) | MoE | fused_topk + moe_align_block_size + 单 kernel batched expert GEMM |
| 13 | [模型加载与权重切分](./13-模型加载与权重切分.md) | 权重 | safetensors 流式；TP shard；q/k/v merge；MoE expert stack；GQA head 复制 |
| 14 | [自定义 CUDA 与 Triton 内核](./14-自定义CUDA与Triton内核.md) | 内核 | tvm-ffi JIT；store/index/radix 各自的优化点；AOT vs JIT 加载 |
| 15 | [总结与未实现的部分](./15-总结与未实现的部分.md) | 收尾 | 全景设计取舍表；与 nano-vllm / 工业 SGLang 对比；扩展方向 |
| 📚 | [参考文献](./references.md) | 论文 | mini-sglang 用到的所有论文清单 + 推荐阅读顺序（Orca / vLLM / SGLang / FA / SARATHI / Megatron-LM / Mixtral / GQA …） |

## 学习路径建议

```
基础动机 (0)
    ↓
分布式架构 (1, 2)        ── 控制流：消息怎么走
    ↓
数据基石 (3, 4, 5)       ── 状态：每个对象代表什么
    ↓
调度核心 (6, 7)          ── 决策：每步选哪个 batch、怎么和 GPU 重叠
    ↓
执行细节 (8, 9, 10)      ── 计算：forward 在做什么
    ↓
分布式与扩展 (11, 12, 13) ── TP / MoE / 权重
    ↓
底层 (14)                ── 自定义 kernel
    ↓
全景对比 (15)
```

**节奏建议**：
- **目标 1 天通读**：每章 30-45 分钟，跳过检查清单，回到代码对照。
- **目标 1 周深读**：每章读完做检查清单，被卡的题回去找代码看，画时序图。
- **目标 2 周精读**：上面 + 自己写一遍 mini 实现（哪怕不能运行，写出框架），对每个设计取舍提一个备选方案。

## 文档使用方式

- 检查清单的折叠答案（`<details>`）在 GitHub / VSCode preview / 多数 markdown 渲染器里点击展开。
- 每章末尾的"下一章预告"承前启后，建议顺序读不要跳。
- 文档里的代码引用都是仓库相对路径（如 [`scheduler/scheduler.py:120-131`](../../python/minisgl/scheduler/scheduler.py)），可以一边读一边跳到源码对照。

## 致谢

文档基于 mini-sglang 开源代码（commit at the time of writing）和 SGLang 项目设计哲学整理。如果发现讲解错误或不够清楚的地方，欢迎在仓库提 Issue 或 PR。
