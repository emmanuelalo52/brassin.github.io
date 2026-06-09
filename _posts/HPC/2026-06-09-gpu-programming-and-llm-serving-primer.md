---
layout: post
title: "GPU Programming & LLM Serving: A Practical Primer"
date: 2026-06-09
category: HPC
author: Emmanuel Alo
excerpt: "From silicon to serving: GPUs, the execution model, GPU DSLs and Triton, CUDA Graphs, FlashInfer attention, and vLLM."
---

This primer walks the full stack of modern LLM inference, from the silicon up to a production serving engine. We start with why a GPU is shaped the way it is, climb through the execution and memory model, look at how kernels get written (CUDA and Triton), and finish with the systems that stitch those kernels into a server: CUDA Graphs, FlashInfer, and vLLM.

{% include toc.html %}

![GPU Programming and LLM Serving title slide](/images/articles/gpu-llm-serving/slide-01.png)

## The GPU: Parallel by Design

![CPU versus GPU core design](/images/articles/gpu-llm-serving/slide-02.png)

A CPU has a few powerful cores tuned for latency. A GPU flips the trade-off: thousands of simpler cores that run the same operation across huge batches of data at once. That throughput model is exactly what dense tensor math wants.

The headline numbers tell the story:

- **10,000s** of parallel cores on a modern GPU
- **TB/s** of on-chip memory bandwidth
- **SIMT**: single instruction, many threads {% include sidenote.html text="SIMT is NVIDIA's spin on SIMD. Threads are programmed independently but executed in lockstep groups, which is what lets you write scalar-looking code that still runs as a wide vector operation." %}

## Threads, Warps & the CTA

![Thread, warp, CTA, and grid hierarchy](/images/articles/gpu-llm-serving/slide-03.png)

Work on a GPU is organized as a hierarchy. You write one program, and the hardware runs it across many threads, grouped into units that share resources and can cooperate.

- **Thread** is one stream of execution, the smallest unit of work.
- **Warp** is 32 threads that execute in lockstep (SIMT). It is the scheduling unit.
- **CTA / Block** is a Cooperative Thread Array: threads that share fast on-chip shared memory and can synchronize. A block runs on one SM.
- **Grid** is all the blocks launched for a single kernel. It spans the GPU.

Why the CTA matters: shared memory plus barriers inside a block are what make fast kernels possible. Tiling a matmul, staging data, and reducing across threads all depend on this cooperative layer.

## The Memory Hierarchy

![On-chip versus off-chip GPU memory tiers](/images/articles/gpu-llm-serving/slide-04.png)

A GPU has tiers of memory that trade speed for size. The closer to the compute cores, the faster and smaller. Performance is mostly decided by which tier your data lives in.

| Tier | Scope | Latency / size |
| ---- | ----- | -------------- |
| Registers | per-thread, on-chip | ~1 cycle, KBs/thread |
| Shared Mem / L1 | per-block (SM), managed | ~20-30 cycles, ~100-228 KB/SM |
| L2 Cache | shared across all SMs | ~200 cycles, tens of MB |
| Global Mem (HBM) | whole GPU, off-chip | ~400+ cycles, GBs @ TB/s |

The kernel engineer's job is to move data up the hierarchy and keep it there: coalesce global loads, stage tiles in shared memory, and hold accumulators in registers. FlashAttention is the canonical example. It keeps the entire attention computation in on-chip SRAM instead of round-tripping scores through HBM. {% include sidenote.html text="Dao et al. (2022) reformulate attention as a tiled, online-softmax computation that never materializes the full N x N score matrix in HBM, turning a memory-bound op into a compute-bound one." ref="https://arxiv.org/abs/2205.14135" ref_title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" %}

## GPGPU & CUDA

![Host and device model in CUDA](/images/articles/gpu-llm-serving/slide-05.png)

GPGPU means using the GPU's thousands of cores for general computation, not just graphics. CUDA is NVIDIA's platform and C++ language for exactly that: you write kernels that run across the thread / block / grid hierarchy, while the CPU orchestrates.

The mental model is host and device. The **CPU (host)** launches kernels and moves data. The **GPU (device)** runs the grid of blocks in parallel. One source file spans two worlds: `__global__` functions run on the device, and the rest runs on the host.

What CUDA is used for in practice:

- **Kernel engineering**: hand-write fast kernels (GEMM, attention, norms, fused elementwise) that exploit tensor cores, async copy, and shared memory.
- **Inference engines**: an LLM forward pass is a sequence of CUDA kernels. Engines assemble, schedule, and reuse them across requests.

## What is a DSL?

![Domain-specific languages and why they matter for GPUs](/images/articles/gpu-llm-serving/slide-06.png)

A Domain-Specific Language trades generality for power inside one narrow problem. Instead of expressing everything from scratch, you describe what you want and the compiler handles the messy how. Familiar examples are SQL for querying data, RegEx for text matching, and HTML / CSS for documents and layout.

Why it matters for GPUs: writing raw CUDA is powerful but slow to iterate. You manage threads, shared memory, and synchronization by hand. A GPU DSL lets you write high-performance kernels at a higher level and leans on a compiler to emit efficient GPU code. That DSL is Triton.

## Triton: GPU Kernels in Python

![Triton kernel example in Python](/images/articles/gpu-llm-serving/slide-07.png)

Triton (from OpenAI) lets you write GPU kernels in plain Python. You get close-to-CUDA performance without hand-managing the lowest-level details. The compiler handles thread mapping, memory coalescing, and scheduling. {% include sidenote.html text="Tillet et al. introduced Triton as a tile-based IR and compiler, so the programmer reasons about blocks of data while the compiler owns intra-tile parallelism and memory access patterns." ref="https://openai.com/index/triton/" ref_title="OpenAI: Introducing Triton" %}

- **Pythonic**: decorate a function with `@triton.jit`, no separate C++ toolchain.
- **Auto-tuned**: the compiler handles coalescing, scheduling, and register allocation.
- **Fast to iterate**: fuse ops and prototype custom kernels in minutes, not days.

A complete vector-add kernel looks like this:

```python
@triton.jit
def add(x, y, out, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x + offs, mask=mask)
    b = tl.load(y + offs, mask=mask)
    tl.store(out + offs, a + b, mask=mask)
```

## How Triton Thinks in Blocks

![Triton block and tile programming model](/images/articles/gpu-llm-serving/slide-08.png)

CUDA makes you reason about individual threads. Triton raises the unit of work to a block (tile): you load, compute on, and store whole chunks of data at once. The compiler maps that tile onto threads for you.

Picture a 1-D tensor split into BLOCK-sized tiles, with one program instance per tile (`pid = 0, 1, 2, ...`). The three primitives you lean on:

- `tl.program_id` answers "which tile am I?" and gives you your slice of the data.
- `tl.arange` plus a `mask` builds the index range and masks off the ragged tail.
- `tl.load` / `tl.store` move a whole tile to and from global memory.

## CUDA Graphs: Capture Once, Replay

![CUDA Graphs reduce per-launch overhead](/images/articles/gpu-llm-serving/slide-09.png)

Every kernel launch carries CPU-side overhead. In an LLM decode loop you launch the same sequence of small kernels thousands of times, and that overhead starts to dominate. CUDA Graphs let you record a sequence once and replay it as a single unit.

Without graphs, the CPU pays launch overhead before every kernel. With a graph, one replay fires the kernels back-to-back. What you get:

- **Lower per-step latency**: the CPU stops being the bottleneck in tight loops.
- **Tighter latency variance**: fewer launch stalls means steadier tokens/sec.
- **Ideal for decode**: the loop is fixed-shape and repeated, the perfect graph candidate.

## FlashInfer: Attention for Serving

![FlashInfer attention kernels for serving](/images/articles/gpu-llm-serving/slide-10.png)

Attention is the hot path of LLM inference. FlashInfer is a library of highly optimized attention kernels built specifically for serving: handling the paged KV cache, wildly varying sequence lengths, and both the prefill and decode phases. {% include sidenote.html text="FlashInfer ships block-sparse and composable attention templates with a JIT layer, so engines can specialize kernels per request shape instead of paying for one rigid implementation." ref="https://arxiv.org/abs/2501.01005" ref_title="FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving" %}

- **Paged KV cache**: reads attention directly over non-contiguous, paged KV blocks.
- **Ragged batches**: efficient when every request in a batch has a different length.
- **Prefill + decode**: specialized kernels for the long prefill and the token-by-token decode.
- **Plugs into engines**: used as the attention backend by serving stacks like vLLM and SGLang.

## Anatomy of an Inference Engine

![Anatomy of an inference engine forward pass](/images/articles/gpu-llm-serving/slide-11.png)

A model's forward pass is a pipeline of GPU kernels. An inference engine wires them together, then schedules and reuses that pipeline across many requests. The forward pass runs:

> Tokens → Embedding → Transformer (× N layers) → Final Norm → LM Head → Sample

That pipeline is built from three groups of kernels and systems:

- **Compute kernels**: GEMM / matmul for QKV and MLP projections (cuBLAS, CUTLASS, Triton); attention for prefill and decode (FlashInfer); fused RMSNorm and SwiGLU for norm and activation.
- **Memory**: a paged KV cache reused per request, weight layout kept resident in HBM, and shared-memory tiling inside every kernel.
- **Orchestration**: a scheduler doing continuous batching of requests, CUDA Graphs to replay the decode loop, and sampling via softmax and top-k / top-p.

## vLLM: High-Throughput Serving

![vLLM serving engine and PagedAttention](/images/articles/gpu-llm-serving/slide-12.png)

vLLM ties the stack together into a production inference server. Its key idea, PagedAttention, manages the KV cache like virtual memory, so the GPU stays busy across many concurrent requests. {% include sidenote.html text="Kwon et al. (2023) borrow paging from operating systems: the KV cache is split into fixed-size blocks that need not be contiguous, which nearly eliminates fragmentation and lets many requests share GPU memory efficiently." ref="https://arxiv.org/abs/2309.06180" ref_title="Efficient Memory Management for Large Language Model Serving with PagedAttention" %}

- **PagedAttention**: KV cache in fixed pages, so almost no memory is wasted.
- **Continuous batching**: requests join and leave the batch every step.
- **High throughput**: many more tokens/sec per GPU under load.

Where it shows up:

- **Chat and assistants**: low-latency streaming responses to many users at once.
- **Model APIs**: an OpenAI-compatible endpoint behind a hosted LLM product.
- **Batch inference**: offline scoring, summarization, or data labeling at scale.

## It All Stacks Up

![The full GPU and LLM serving stack](/images/articles/gpu-llm-serving/slide-13.png)

Reading from the lowest level up to the API, the whole picture stacks like this:

| Layer | Role |
| ----- | ---- |
| vLLM | serving engine: batching, scheduling, the API |
| FlashInfer | optimized attention kernels on the hot path |
| CUDA Graphs | replay the decode loop with minimal overhead |
| Triton / DSL | custom fused kernels, written productively |
| CUDA / GPGPU | the platform: kernels across the thread hierarchy |
| Threads & CTAs | the cooperative execution and memory model |
| GPU | thousands of parallel cores |

Each layer exists to keep the one above it fed and the silicon below it busy. Understand the memory hierarchy and the execution model at the bottom, and every optimization higher up (fused kernels, graph replay, paged attention, continuous batching) reads as the same idea applied at a different scale: keep the compute units saturated and stop paying for data movement you do not need.
