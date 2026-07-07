---
title: "LLM Inference Lab"
date: 2026-06-22
type: "project"
section: "Projects"
category: "LLM Inference"
tags: ["LLM Inference", "Benchmark", "AI infra"]
concepts: ["KV Cache", "Benchmarking", "Prefill", "Decode", "Serving Runtime"]
parents: ["AI Systems Project", "LLM Inference"]
papers: ["vllm-pagedattention"]
summary: "一个用于记录 LLM 推理系统实验、profiling 方法和 serving 优化结论的项目归档示例。"
status: "active"
repoUrl: "https://github.com/xielinrui/llm-inference-lab"
---

## 项目目标

建立一套可复现实验，用于比较不同推理后端在延迟、吞吐、显存利用率上的表现。

## 当前范围

- Prompt / decode 分阶段 profiling
- 不同 batch size 和 sequence length 的吞吐曲线
- KV Cache 与调度策略观察

## 实验记录

后续每次实验建议记录：硬件、模型、后端版本、参数、数据集、命令和结果摘要。

## 结论

项目归档页用于保留工程背景、实验上下文和阶段性判断，避免结果散落在临时脚本中。
