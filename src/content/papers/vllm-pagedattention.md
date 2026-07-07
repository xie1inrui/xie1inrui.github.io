---
title: "vLLM: PagedAttention 论文阅读"
date: 2026-06-22
type: "paper"
section: "Research"
category: "LLM Inference"
tags: ["LLM Inference", "Serving", "PagedAttention", "AI System"]
concepts: ["KV Cache", "PagedAttention", "Continuous Batching", "Memory Fragmentation", "Prefix Caching"]
parents: ["Paper Reading", "LLM Inference"]
projects: ["LLM Inference Lab"]
summary: "从 KV Cache 管理角度理解 vLLM 的 PagedAttention 如何提升吞吐和显存利用率。"
status: "reading"
authors: ["Woosuk Kwon", "Zhuohan Li", "Siyuan Zhuang"]
venue: "SOSP"
year: 2023
paperUrl: "https://arxiv.org/abs/2309.06180"
codeUrl: "https://github.com/vllm-project/vllm"
---

## 论文信息

- 论文：Efficient Memory Management for Large Language Model Serving with PagedAttention
- 主题：LLM Serving / KV Cache / Continuous Batching

## 研究问题

LLM Serving 中 KV Cache 占用大、生命周期动态变化，传统连续显存分配容易造成碎片和浪费。

## 核心思想

借鉴操作系统虚拟内存分页思想，把 KV Cache 切成固定大小 block，通过 block table 做逻辑 token 到物理块的映射。

## 系统架构

- Scheduler：管理请求队列和 continuous batching
- Block Manager：分配、回收、共享 KV blocks
- PagedAttention Kernel：按 block table 读取非连续 KV Cache

## 方法细节

1. 固定大小 KV block 降低外部碎片。
2. Prefix sharing 通过引用计数复用已有 block。
3. Copy-on-write 支持 beam search 等分支场景。

## 实验设置

关注吞吐、延迟、显存利用率和不同请求长度分布下的稳定性。

## 结果分析

系统收益主要来自显存浪费减少，使同等 GPU 显存能容纳更多并发序列。

## 优缺点

优点：抽象清晰，系统收益显著，适合真实 serving workload。缺点：kernel 与调度系统复杂度增加。

## 我的理解

PagedAttention 的价值不是单个 kernel 优化，而是把内存管理从“张量连续分配”提升为 serving runtime 的一等抽象。

## 可借鉴点

- 把 OS 经典抽象迁移到 AI infra。
- 用 workload 特征指导系统设计。

## 未解决问题

- 多机多卡场景下 KV Cache paging 与通信调度如何协同？
- 长上下文、多租户 serving 下是否需要分层 KV 存储？
