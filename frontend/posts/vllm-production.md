---
title: "Deploying vLLM in Production"
description: "Notes from taking a vLLM serving stack from a laptop demo to a real endpoint."
date: "2026-09-10"
updated: "2026-09-12"
category: "MLOps"
tags: ["MLOps", "Backend", "DevOps", "vLLM"]
status: "published"
featured: true
readingTime: 9
---

## Why vLLM

Continuous batching and paged attention gave the biggest throughput win over a naive HF Transformers server.

```mermaid
graph LR
A[Client] --> B[Load Balancer]
B --> C[vLLM Pod 1]
B --> D[vLLM Pod 2]
```

## Rollout

Health checks needed to account for warm-up latency on the first request after a cold container start.
