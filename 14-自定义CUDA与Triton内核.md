# 第 14 章：自定义 CUDA 与 Triton Kernel

> Mini-SGLang 内置了一些自定义 kernel，覆盖了 PyTorch 算子链效率不够、又不属于"通用 attention"的场景：`store_cache` / `indexing` / `fast_compare_key` / `fused_moe_kernel_triton` / `pynccl`。
>
> 这一章讲清楚：
> 1. 这些 kernel 各自解决什么问题、为什么不能用 PyTorch 直接写。
> 2. mini-sglang 怎么用 **tvm-ffi** 做 JIT 编译加载。
> 3. AOT vs JIT 两种模式的差异。
>
> 入口：[`kernel/`](../../python/minisgl/kernel/)，C++/CUDA 源码在 [`kernel/csrc/`](../../python/minisgl/kernel/csrc/)。

---

## 14.1 五个内置 kernel 的总览

| Kernel | 文件 | 用在哪 | 为什么自定义 |
|--------|------|------|-------|
| `store_cache` | [`kernel/store.py`](../../python/minisgl/kernel/store.py) + `csrc/src/store.cu` | `MHAKVCache.store_kv` | element_size 模板化，多 dtype/head_dim 一份代码 |
| `indexing` | [`kernel/index.py`](../../python/minisgl/kernel/index.py) + `csrc/src/index.cu` | `VocabParallelEmbedding.forward` | vocab_range mask + 大行 split 优化 |
| `fast_compare_key` | [`kernel/radix.py`](../../python/minisgl/kernel/radix.py) + `csrc/src/radix.cpp` | `RadixTreeNode.get_match_len` | CPU 端高速比较两个 1D int tensor |
| `fused_moe_kernel_triton` | [`kernel/triton/fused_moe.py`](../../python/minisgl/kernel/triton/fused_moe.py) | `FusedMoe.forward` | sorted token + batched expert GEMM |
| `pynccl` | [`kernel/pynccl.py`](../../python/minisgl/kernel/pynccl.py) + `csrc/src/pynccl.cu` | `LinearOProj.forward` 等 | 绕开 torch.distributed.nccl 的隐式同步 |

下面逐个看。

---

## 14.2 tvm-ffi 加载机制

mini-sglang 的 C++/CUDA kernel 通过 `tvm-ffi` 加载——不用 PyBind11 / pybind / torch 的扩展机制。两个原因：
1. **更轻**：tvm-ffi 是 TVM 团队的 FFI 库，没有 PyTorch 扩展那么多依赖。
2. **JIT 编译快**：`load_inline` 风格的字符串拼接，热启动后只跑 `nvcc` 一遍。

[`kernel/utils.py`](../../python/minisgl/kernel/utils.py) 提供两种加载模式：

### `load_aot`：编译已有源文件

```python
def load_aot(*args, cpp_files=None, cuda_files=None, ...):
    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]
    return tvm_ffi.cpp.load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        ...
    )
```

直接编译磁盘上的 .cpp / .cu 文件，得到一个 .so 加载进来。

用例：`fast_compare_key`（[`radix.py:14-15`](../../python/minisgl/kernel/radix.py)）只是个 cpp 函数，AOT 编译一次就够。pynccl 也是 AOT。

### `load_jit`：模板特化时拼源码再编译

```python
def load_jit(*args, cuda_files=None, cuda_wrappers=None, ...):
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
    return tvm_ffi.cpp.load_inline(
        _make_name(*args),
        cuda_sources=cuda_sources,    # ← string, not file
        ...
    )
```

它接受 `(export_name, kernel_name)` 元组列表，自动生成包装代码：

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, (StoreKernel<256, 1, false>::run));
```

把模板类 `StoreKernel<element_size, num_threads, max_occupancy, use_pdl>` 实例化成具体的全局函数，导出给 Python 调用。

每次 element_size 不同，会生成不同的实例化代码、调一次 `nvcc`，结果被 [`functools.cache`](https://docs.python.org/3/library/functools.html) 包住，下次同 element_size 直接复用。

> 第一次启动时 nvcc 编译 6-7 个 kernel 大概要 30-60s，后续启动（缓存命中）几秒就完事。如果改了源码或换了 nvcc 版本、CUDA 版本，缓存自动失效重编。

### `KernelConfig`：模板参数管理

```python
class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool
```

**PDL = Programmatic Dependent Launch**，是 Hopper (SM90) 引入的硬件特性（[NVIDIA CUDA Programming Guide §7.30 Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)）。

朴素理解：CUDA 默认行为下，kernel B 必须等 kernel A 的**所有 SM 都退出**才能开始 launch（grid prologue）。PDL 让 kernel B 可以**在 kernel A 还有一些 block 没退出时就开始执行 grid prologue**（分配 shared memory、加载常量、issue 早期 load 指令），prologue 与 kernel A 的 epilogue 重叠。两者通过显式的 `cudaTriggerProgrammaticLaunchCompletion()`（在 kernel A 末尾调）+ `cudaGridDependencySynchronize()`（在 kernel B 开头调）协调——即生产者 / 消费者关系是显式的。

**适用场景**：连续两个 kernel，第二个的 prologue 较重（比如要加载较大常量到 shared memory）。在 LLM 推理里 PDL 收益不大（每个 kernel 自己就足够大），mini-sglang 默认 `use_pdl=False`。FA3 / FA4 用了 PDL 来加速连续 layer 的 attention kernel 衔接。

`make_cpp_args(element_size, num_threads, max_occupancy, use_pdl)` 把 Python 参数翻译成 C++ 模板参数字符串。

---

## 14.3 store_cache：为什么不用 PyTorch

[`kernel/store.py`](../../python/minisgl/kernel/store.py)：

```python
def store_cache(k_cache, v_cache, indices, k, v):
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
```

效果是 `k_cache[indices[i]] = k[i]; v_cache[indices[i]] = v[i]` 并行做。

为什么不用 PyTorch？

```python
# 朴素 PyTorch 写法
k_cache[indices] = k
v_cache[indices] = v
```

这样的问题：
- 启动 2 个 kernel（一个 K、一个 V），launch 开销大。
- 每个 kernel 的 element 拷贝带宽利用不充分（PyTorch 的 advanced indexing 走的是通用 path，没针对"连续几个字节一组"做优化）。

mini-sglang 的 `StoreKernel` 模板化 `element_size`（一行字节数），针对每种 head_dim × num_kv_heads × dtype 组合编译一份特化版，向量化 load/store。一个 kernel 同时写 K 和 V。

---

## 14.4 indexing：embedding 的 vocab_range 优化

[`kernel/index.py`](../../python/minisgl/kernel/index.py) 给 `VocabParallelEmbedding` 用：

```python
def indexing(weights, indices, *, output=None, vocab_range=None):
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:    num_splits = 4
    elif element_size % 1024 == 0:  num_splits = 2
    else:                            num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
```

效果：`output[i] = weights[indices[i] - start] if indices[i] in [start, start+length) else 0`。

`vocab_range = (start, length)` 是 TP 切完后这个 rank 持有的 vocab 范围。Token id 落在范围外的，输出 0（之后 all_reduce 时被加起来 = 完整 embedding）。

朴素 PyTorch 写法需要 mask + scatter：

```python
mask = (indices >= start) & (indices < start + length)
local_indices = (indices - start).clamp(min=0, max=length-1)
out = torch.zeros(indices.shape[0], hidden_dim)
out[mask] = weights[local_indices[mask]]
```

慢且分散——多次 launch、不必要的中间张量。自定义 kernel 一行代码做完，还能根据 element_size 决定 `num_splits`：行数据特别大时（几 KB）拆成多个 thread block 处理同一行，提升带宽利用。

---

## 14.5 fast_compare_key：CPU 端的最长公共前缀

[`kernel/radix.py`](../../python/minisgl/kernel/radix.py)：

```python
def fast_compare_key(x, y):
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
```

输入两个 1D int tensor（CPU 上），返回它们的最长相同前缀长度。给 [`RadixTreeNode.get_match_len`](../../python/minisgl/kvcache/radix_cache.py:63-67) 用：

```python
def get_match_len(self, input_ids):
    from minisgl.kernel import fast_compare_key
    return fast_compare_key(self._key, input_ids)
```

Radix tree 每次走一层就要比较一条边的所有 token——朴素 Python `for i in range(len(key)): if key[i] != input_ids[i]: break` 慢得要命（GIL + 解释器循环），而 `(key == input_ids).cumprod()` 这种 PyTorch 写法又过度（分配大 tensor）。

C++ 版本简单直接：

```cpp
// kernel/csrc/src/radix.cpp（伪代码）
int fast_compare_key(Tensor x, Tensor y) {
    auto* xp = x.data_ptr<int32_t>();
    auto* yp = y.data_ptr<int32_t>();
    int n = std::min(x.numel(), y.numel());
    for (int i = 0; i < n; ++i)
        if (xp[i] != yp[i]) return i;
    return n;
}
```

实际实现可能用 SIMD（AVX2/AVX-512）批量比较加速。

为什么是 CPU kernel 而不是 GPU？因为 RadixCache 的 `_tree_walk` 在 scheduler 主循环里跑，相对小数据（一条边的 token 数百到几千），CPU 端单线程 + SIMD 能在几微秒内做完——**比把数据传到 GPU、launch kernel、再传回来快得多**。

---

## 14.6 fused_moe_kernel_triton：用 Triton 写

第 12 章详细讲过逻辑，这里说**为什么用 Triton 而不是 CUDA**：

[`kernel/triton/fused_moe.py`](../../python/minisgl/kernel/triton/fused_moe.py) 是 Triton kernel——Python DSL 编译成 PTX。用 Triton 的好处：

1. **可读性高**：MoE kernel 有复杂的 grid、tile、padding 逻辑，CUDA 写至少几百行；Triton 几十行就能搞定。
2. **autotune**：Triton 内置自动调 BLOCK_SIZE。CUDA 版需要手工 grid search。
3. **性能不输 CUDA**：对 MoE 这种"稠密 GEMM 包了一层动态分发"的场景，Triton 的优化器很擅长。

代价：Triton 第一次运行有 JIT 编译开销（几百毫秒到几秒）；不能直接进 CUDA Graph capture（需要 Triton 配合 capture 友好的 launch 方式——mini-sglang 用的是 fused_moe 已经是 capture-friendly 的版本）。

---

## 14.7 pynccl：从底层 NCCL 直接绑定

[`kernel/pynccl.py`](../../python/minisgl/kernel/pynccl.py) 第 11 章已详细讲。这里只看加载部分：

```python
@functools.cache
def _load_nccl_module():
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])

@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi
    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)
    return PyNCCLImpl
```

两步：
1. **`load_aot`** 编译 `pynccl.cu`（链接 `-lnccl`）成 .so。这里 `extra_ldflags` 显式指定要链接 NCCL 库。
2. **`@tvm_ffi.register_object`** 把 C++ 的 `minisgl.NCCLWrapper` 类注册成 Python 类。`__ffi_init__` 调对应的 C++ 构造函数。

这种"C++ 类 ↔ Python 类"的 binding 比 PyBind11 简单，但不如它通用——只用在 mini-sglang 这种集中式自定义 kernel 的场景合适。

---

## 14.8 一个 kernel 加载流程

```mermaid
flowchart LR
    py["Python 调用<br/>minisgl.kernel.xxx"] --> loader{"加载方式"}
    loader -->|load_aot| cu["已有 .cu / .cc 源文件"]
    loader -->|load_jit| tmpl["Python 模板拼源码"]
    cu --> build["tvm-ffi 编译 / 加载 .so"]
    tmpl --> build
    build --> reg["注册 FFI 函数 / Object"]
    reg --> call["Python 直接调用 kernel wrapper"]
    call --> gpu["CUDA / Triton kernel<br/>在当前 stream 执行"]
```

跟踪一下 `store_cache` 第一次被调用时的全链：

```
Engine.__init__ 
 → 模型 forward
 → AttentionLayer.forward
 → ctx.attn_backend.forward (FA backend)
 → self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
 → store_cache(k_cache, v_cache, indices, k, v)
   ├─ k_cache.view(num_tokens, -1)               # 折叠成 (T, head*dim)
   ├─ element_size = k_cache.shape[1] * 2        # bf16 = 2 bytes
   ├─ _jit_store_module(element_size)             # @functools.cache 包着
   │   └─ load_jit("store", "<element_size>", ..., 
   │              cuda_files=["store.cu"],
   │              cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")])
   │       └─ tvm_ffi.cpp.load_inline:
   │           # 拼字符串：
   │           # #include ".../jit/store.cu"
   │           # TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, (StoreKernel<256, 128, 1, false>::run));
   │           # 调 nvcc 编译，得到 minisgl__store__256_128_1_false.so
   │           # 加载到 Python，返回 Module
   └─ module.launch(k_cache, v_cache, indices, k, v)
       # 实际就是调 StoreKernel<...>::run(...) 这个 CUDA kernel
```

第二次调用同 element_size：`@functools.cache` 命中，直接拿到同一个 module，绕过 nvcc，几十纳秒。

---

## 14.9 kernel 源码结构

```
python/minisgl/kernel/
├── __init__.py        ← 导出 indexing / store_cache / fast_compare_key 等
├── index.py           ← Python 包装
├── store.py
├── radix.py
├── pynccl.py
├── tensor.py          ← 测试用的 tensor 工具
├── moe_impl.py        ← Triton MoE 入口
├── utils.py           ← load_aot / load_jit
├── triton/
│   └── fused_moe.py   ← Triton kernel
└── csrc/
    ├── include/
    │   └── minisgl/
    │       └── *.h    ← 共享 header
    ├── src/
    │   ├── store.cu (?)
    │   ├── radix.cpp
    │   └── pynccl.cu
    └── jit/
        └── store.cu   ← JIT 模板源码
```

`csrc/include/minisgl/nccl227.h` 在 [`pyproject.toml`](../../pyproject.toml) 里被排除出 clang-format（[CLAUDE.md 提到](../../CLAUDE.md)）——它是 NCCL 头文件的本地副本（适配特定版本），手动维护格式。

---

## 14.10 调试与扩展

### 看编译后的 .so

`tvm-ffi` 默认把编译产物放在 `~/.cache/tvm-ffi/`。失败时可以去那里看 nvcc 的输出和生成的 wrapper 源码。

### 强制重编

```bash
rm -rf ~/.cache/tvm-ffi/minisgl__store__*
```

### 加新 kernel 的步骤

1. 在 `csrc/src/foo.cu` 写 CUDA kernel + 模板 wrapper。
2. 在 `kernel/foo.py` 写 Python 包装：
   ```python
   @functools.cache
   def _jit_foo_module(...):
       return load_jit("foo", *args, cuda_files=["foo.cu"], cuda_wrappers=[("launch", f"FooKernel<{args}>::run")])

   def foo(...):
       module = _jit_foo_module(...)
       return module.launch(...)
   ```
3. 在 `kernel/__init__.py` 里 export。
4. 调用方 `from minisgl.kernel import foo`。

### 生产环境下避免 JIT

`load_jit` 在用户首次启动时编译——首次延迟 30-60s 不可接受。生产部署时常见做法：

- **预热**：服务启动后跑一次 dummy forward，让所有 JIT 编译完成。
- **缓存复用**：`build_directory` 参数指定持久缓存目录，跨容器重启复用编译结果。

---

## 14.11 检查清单

1. **`load_aot` 和 `load_jit` 的本质区别？什么场景该用哪个？**
   <details><summary>参考答案</summary>

   - **`load_aot`**：编译磁盘上现成的 .cpp/.cu 文件，没有模板特化，一份代码一个 .so。
   - **`load_jit`**：拼字符串 → `load_inline`，把模板参数（element_size、threads 等）写成宏或模板值，按参数生成不同 .so。

   场景：
   - 函数签名固定、不需要按运行时参数特化 → AOT（如 `fast_compare_key`、`pynccl.cu`）。
   - 函数有"按 hidden_size 一组"的特化（如 `store_cache` 不同 head_dim 一份代码） → JIT。
   </details>

2. **`@functools.cache` 在 `_jit_store_module` 上是干什么用的？删了会怎样？**
   <details><summary>参考答案</summary>

   缓存 (element_size, config) → Module 的映射。

   删了之后：每次调用 `store_cache` 都会触发一次 `load_jit` → `nvcc` 编译——几十秒一次。一次 forward 30 层 attention，30 次编译，启动直接卡死。

   `@functools.cache` 让相同参数的编译只跑一次，结果在内存里持久缓存（进程生命周期内）。
   </details>

3. **为什么 `fast_compare_key` 跑在 CPU 上而不是 GPU？**
   <details><summary>参考答案</summary>

   - **数据量小**：每次比较一条 RadixTree 边的 token 序列（几十到几千 int32）。GPU launch 开销几十微秒，比 CPU 跑完的时间还长。
   - **数据已经在 CPU**：`RadixTreeNode._key` 是 CPU tensor，`input_ids` 也是 CPU tensor。GPU 跑要先 H2D 再 D2H，反而慢。
   - **scheduler 主循环在 CPU 端**：tree_walk 是同步逻辑，把它放 GPU 又要等结果，得不偿失。

   CPU 端用 SIMD（AVX-512 一次比 16 个 int）能在 1-2 微秒内比完几千个 int——足够快。
   </details>

4. **`fused_moe_kernel_triton` 用的是 Triton 而不是 CUDA。Triton 的 JIT 第一次跑有几秒延迟，会影响 mini-sglang 启动吗？**
   <details><summary>参考答案</summary>

   会。第一次 forward 时 Triton 会触发 JIT 编译，单次 100ms ~ 几秒不等。

   缓解方法：
   - **Engine 启动时 warmup**：[`graph.py:_capture_graphs:138-139`](../../python/minisgl/engine/graph.py)) 在 capture 之前先跑一次 dummy forward——这次 forward 顺便触发了 fused_moe 的 JIT 编译。
   - **第一次真实请求依然有 JIT**：因为 dummy 用的是 max_graph_bs，第一次真实请求可能用更小 bs，Triton 会按 bs autotune 重新生成代码。可以在 capture 时枚举 bs_list 都跑一遍 dummy 触发 autotune。

   生产部署：建议预热 + persist Triton 的 cache（默认在 `~/.triton/cache`）。
   </details>

5. **如果你想让 mini-sglang 在没有 sgl_kernel 的环境里跑（比如老 PyTorch），最少要改哪些代码？**
   <details><summary>参考答案</summary>

   sgl_kernel 提供：
   - `flash_attn_with_kvcache`（FA 后端）
   - `topk_softmax`（fused_topk）
   - `moe_align_block_size`（MoE 排序）

   替换路径：
   - `flash_attn_with_kvcache` → 用 FlashInfer backend 替代（mini-sglang 的 fi 后端不依赖 sgl_kernel）。CLI 加 `--attn fi`。
   - `topk_softmax` → 改成 `gating.softmax(-1).topk(top_k)` 的 PyTorch 版本，慢一些但能跑。
   - `moe_align_block_size` → 写一个 Triton 版本（开源代码不少），或者放弃 fused MoE 走 per-expert 循环。

   实际上 sgl_kernel 是 mini-sglang 的硬依赖（`pyproject.toml` 写明），项目定位是"Linux + NVIDIA GPU"，没考虑 sgl_kernel 缺失场景。如果一定要去掉，就放弃 FA 后端 + 改 fused MoE，工程量不小。
   </details>

---

## 下一章预告

最后一章是总结。我们把 mini-sglang 的所有设计取舍画成一张大表，对照 nano-vllm 和工业 SGLang 各自缺什么/有什么，给出后续的学习方向（EP、speculative decoding、KV offload 等）。
