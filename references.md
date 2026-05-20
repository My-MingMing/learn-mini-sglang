# 参考文献与延伸阅读

> 本文档汇总所有章节用到的论文 / 资料 / 标准实现。每篇按"它是 mini-sglang 哪部分代码的设计源头"展开，方便回头追根溯源。
>
> 引用风格：作者 (年份)，arXiv id 或会议名。**所有论文链接是公开预印本**，部分可能与最终版本略有差异。

---

## 1. 推理引擎与系统调度

### Orca: A Distributed Serving System for Transformer-Based Generative Models
- **Yu et al., OSDI 2022**
- 提出 **continuous batching**（也叫 iteration-level scheduling）：每个 forward step 重新组 batch，让先完成的请求立刻退出、新的请求立刻加入，而不是等整个 batch 跑完才换。这是现代 LLM 推理引擎的基础。
- mini-sglang 的 `Scheduler.run_forever` 每步都重新成 batch 就是 Orca 思想的最小实现。

### Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)
- **Kwon et al., SOSP 2023, arXiv:2309.06180**
- 提出 **PagedAttention**：把 KV cache 切成固定大小 page、按需分配、用 page table 间接寻址，类比操作系统的虚拟内存。
- mini-sglang 的 [`MHAKVCache`](../../python/minisgl/kvcache/mha_pool.py) + [`CacheManager`](../../python/minisgl/scheduler/cache.py) 的 paged 设计直接来自这篇。

### SGLang: Efficient Execution of Structured Language Model Programs
- **Zheng et al., NeurIPS 2024, arXiv:2312.07104**
- 提出 **RadixAttention**：用 radix tree（基数树）管理 paged KV cache，让相同 prefix 的请求**自动**复用 KV，而不是只对显式标注的"system prompt"做缓存。
- mini-sglang 的 [`RadixPrefixCache`](../../python/minisgl/kvcache/radix_cache.py) 是这个算法的精简实现——树结构、ref_count、LRU 驱逐都直接对应论文 Figure 4。
- 论文 8.2 节实测 RadixAttention 在 chat / multi-turn / few-shot prompt 场景比 vLLM 朴素 paged KV 提升 2-5×。

### SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
- **Agrawal et al., arXiv:2308.16369 (2023)**

### Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- **Agrawal et al., OSDI 2024, arXiv:2403.02310**
- 提出 **chunked prefill + piggyback**：把长 prompt 切成多个 chunk，每个 chunk 跑一次 forward；同一个 batch 里**混入正在 decode 的请求**让 GPU 不空闲。
- mini-sglang 实现了 chunked prefill（[`ChunkedReq`](../../python/minisgl/scheduler/prefill.py) + `--max-prefill-length`），但**没有 piggyback**——它的 batch 是纯 prefill 或纯 decode。这是 mini-sglang 与工业 SGLang 的一个差异。
- DeepSpeed-FastGen 的"Dynamic SplitFuse"是同一思路的另一种实现。

---

## 2. 注意力与 KV cache

### FlashAttention (FA1)
- **Dao et al., NeurIPS 2022, arXiv:2205.14135**
- 把 attention 的 softmax(QKᵀ)V 算成 IO-aware 的 tiled 版本，避免显存里物化 N² 的 attention matrix——这是后来所有 LLM 训练/推理 attention 的标准。

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Dao, arXiv:2307.08691 (2023)**
- 改进 FA1 的并行化，A100 上 230 TFLOPs（FP16）。

### FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
- **Shah et al., arXiv:2407.08608 (2024)**
- 利用 H100 的 **Tensor Memory Accelerator (TMA) + WGMMA** 异步指令，让 GEMM 和 softmax 重叠；FP16 740 TFLOPs，FP8 1.2 PFLOPs。**只支持 SM90+**（Hopper 及之后）。
- mini-sglang [`fa.py:_fa_sgl_impl`](../../python/minisgl/attention/fa.py) 的 `version=3 if SM90 else 4 if SM100` 选择就是基于此（FA4 是为 Blackwell 优化的版本，目前在 sgl_kernel 里实验性提供）。

### FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
- **Ye et al., arXiv:2501.01005 (2025)**
- 一套面向推理（而非训练）的 attention 引擎：plan/run 两阶段、支持 paged KV、支持各种 mask（causal、sliding window、custom）、支持 CUDA Graph capture。
- mini-sglang 的 [`fi.py`](../../python/minisgl/attention/fi.py) 直接调 FlashInfer Python API；TRT-LLM backend 也是通过 FlashInfer 的 trtllm_batch_*_with_kv_cache 子 API 调用 NVIDIA 的 TRT-LLM kernel。

---

## 3. Tensor / Pipeline / Expert Parallelism

### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- **Shoeybi et al., arXiv:1909.08053 (2019)**
- 提出经典的 transformer **column parallel + row parallel** 切法（前者输出维度切，后者输入维度切，组合后只需要 2 次 all_reduce / 层）。
- mini-sglang 的 [`LinearColParallelMerged` / `LinearRowParallel` / `LinearOProj`](../../python/minisgl/layers/linear.py) 命名和数学完全沿用此论文。

### Reducing Activation Recomputation in Large Transformer Models (Sequence Parallelism)
- **Korthikanti et al., MLSys 2023, arXiv:2205.05198**
- 在 TP 之上加 sequence parallel，让 LayerNorm/Dropout 也切 N 路。mini-sglang 没实现这个（推理不需要 recomputation），但工业 SGLang 有。

### GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
- **Lepikhin et al., arXiv:2006.16668 (2020)**
- MoE 的现代起点，提出 top-2 routing + dispatch/combine 算法。

### Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **Fedus et al., arXiv:2101.03961 (2021)**
- 简化为 top-1 routing；提出 expert capacity factor、auxiliary load balancing loss。

### Mixtral of Experts
- **Jiang et al., arXiv:2401.04088 (2024)**
- 第一个开源高质量 MoE 模型；top-2 sparse activation；Qwen3-MoE 直接沿用同一架构思路（top-K, gate softmax + renormalize）。
- mini-sglang [`MoELayer`](../../python/minisgl/layers/moe.py) + [`FusedMoe`](../../python/minisgl/moe/fused.py) 的 routing 逻辑就是这一系列论文的工程实现。

---

## 4. Attention 头数变种（GQA / MQA）

### Fast Transformer Decoding: One Write-Head is All You Need (MQA)
- **Shazeer, arXiv:1911.02150 (2019)**
- 提出 **Multi-Query Attention**：所有 Q head 共享同一组 K/V head（num_kv_heads = 1）。decode 阶段 KV cache 大小缩小 num_qo_heads 倍。

### GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Ainslie et al., EMNLP 2023, arXiv:2305.13245**
- MQA 的折中版：把 num_qo_heads 分成 G 组、每组共享一对 K/V head（`num_kv_heads = num_qo_heads / G`）。Llama-3、Qwen2/3 默认 GQA。
- mini-sglang 的 [`MHAKVCache`](../../python/minisgl/kvcache/mha_pool.py:26-27) 的 `local_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)` 直接处理 GQA：当 TP 数大于 KV head 数时复制 KV head，避免越切越碎。

---

## 5. CUDA Graph 与并发执行

### CUDA Graphs (Programming Guide)
- **NVIDIA CUDA C++ Programming Guide, §3.2.8.7 / §6**
- 把一个或多个 CUDA stream 上的 op 录成 graph，replay 时一次性提交（省去 N 次 cuLaunchKernel 的 syscall 和 driver 开销）。LLM decode 一步要 launch 几百个 kernel，CUDA Graph 在这个场景收益巨大（5-30%）。
- mini-sglang [`GraphRunner`](../../python/minisgl/engine/graph.py) 是标准的"per-batch-size capture / replay-with-buffer-copy"模式。

### Programmatic Dependent Launch (PDL)
- **NVIDIA Hopper architecture (SM90+) feature**
- 让连续两个 kernel 的 prologue/epilogue 在硬件上重叠：当 kernel A 还在跑最后几个 SM 时，kernel B 已经在剩下的 SM 上 setup 自己的 grid（分配 shared memory、加载常量等），减少 kernel-to-kernel 的"接力"延迟。
- mini-sglang 的 `KernelConfig.use_pdl` 控制是否启用（默认 False，因为 PDL 需要相邻 kernel 都 opt-in）。

---

## 6. NCCL 与通信

### NCCL (NVIDIA Collective Communications Library)
- **NVIDIA, [github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)**
- 提供 ring-based all-reduce、tree-based all-reduce、SHARP 加速等 GPU 集体通信原语。
- mini-sglang 通过 [`PyNCCLCommunicator`](../../python/minisgl/kernel/pynccl.py)（一个 tvm-ffi 包装的 C++ class）直接调 NCCL 库，绕开 `torch.distributed.ProcessGroupNCCL`。这样 collective 在调用 stream 上 enqueue（而不是 torch 内部专门的 NCCL stream），与 CUDA Graph capture 兼容。

### NCCL Stream Capture Documentation
- **NCCL ≥2.15** 开始官方支持 CUDA Graph capture（早先版本 capture 时会报 "host function calls inside stream capture" 错误）。
- mini-sglang 要求的 nccl 版本头文件在 [`csrc/include/minisgl/nccl227.h`](../../python/minisgl/kernel/csrc/include/minisgl/) 里 hard-pin。

---

## 7. RoPE 变种（YaRN / llama3 scaling）

### RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Su et al., arXiv:2104.09864 (2021)**
- 原始 RoPE 算法。

### YaRN: Efficient Context Window Extension of Large Language Models
- **Peng et al., arXiv:2309.00071 (2023)**
- 通过频率分段 + 平滑过渡扩展上下文长度（典型 4K → 128K）。
- mini-sglang [`layers/rotary.py`](../../python/minisgl/layers/rotary.py) 的 `case "yarn"` 分支实现。

### Llama 3 RoPE scaling
- **Meta Llama 3 release, 2024**
- 类似 YaRN 但用了不同的 low/high freq factor 划分。
- mini-sglang `case "llama3"` 分支。

---

## 8. 推荐阅读顺序

如果你刚开始研究 LLM 推理引擎，按这个顺序读论文会比较顺：

1. **Orca**（OSDI 2022）— 理解 continuous batching 的动机。
2. **PagedAttention / vLLM**（SOSP 2023）— 理解 KV cache 内存管理。
3. **Megatron-LM TP**（2019）— 理解 transformer TP 数学。
4. **FlashAttention 1**（2022）— 理解 IO-aware attention。
5. **GQA / MQA**（2019/2023）— 理解为什么现代模型 KV head 比 Q head 少。
6. **SGLang RadixAttention**（NeurIPS 2024）— 理解 prefix cache 树结构。
7. **SARATHI / SARATHI-Serve**（2023/2024）— 理解 chunked prefill + piggyback。
8. **FlashAttention 3**（2024）— 理解 Hopper 时代的 attention kernel。
9. **Mixtral**（2024）— 理解 MoE 推理。

读完这 9 篇 + mini-sglang 代码，你对推理引擎的理解会非常完整。

---

## 9. 工业 SGLang 额外有什么

mini-sglang ≈ SGLang 论文 + 一些工程化。**工业 SGLang** ([sgl-project/sglang](https://github.com/sgl-project/sglang)) 在此之上还实现了：

- **Speculative Decoding**：Leviathan et al. 2023 (arXiv:2211.17192) + Chen et al. 2023 (arXiv:2302.01318)
- **EAGLE / EAGLE-2 / EAGLE-3**：一系列改进的 spec decoding 方法
- **Constrained Decoding**：xgrammar / outlines / lm-format-enforcer 集成
- **HiCache**（CPU + Disk KV offload）
- **Disaggregated Prefill/Decode**：把 prefill 和 decode 部署到不同集群
- **Quantization**：AWQ、GPTQ、FP8、INT4 各种 weight + KV 量化
- **MLA (Multi-head Latent Attention)**：DeepSeek 模型的特殊注意力变种
- **Pipeline Parallelism (PP)**
- **Expert Parallelism (EP)** + EP/TP 混合

mini-sglang 的目标不是工程化全集，而是**用最少代码讲清楚核心设计**。想做生产部署应该用工业版。
