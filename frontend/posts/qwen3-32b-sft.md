---
title: "Fine-tuning Qwen3-32B with SFT"
description: "Supervised fine-tuning notes: data prep, LoRA config, and what actually moved eval numbers."
date: "2026-09-18"
updated: "2026-09-18"
category: "LLM"
tags: ["LLM", "SLM", "Fine-tuning", "Qwen"]
status: "published"
featured: true
readingTime: 11
---

## Setup

Fine-tuning a 32B model on a single node meant LoRA plus gradient checkpointing was non-negotiable.

| Setting | Value |
|---|---|
| Rank | 32 |
| Alpha | 64 |
| Target modules | q_proj, v_proj |

## Data

Deduplicating and length-filtering the instruction set mattered more than adding more raw examples.

## Results

Held-out task accuracy improved most from cleaning labels, not from scaling epochs. See [[Understanding RAGAS Evaluators]] for how these checkpoints were scored downstream.
