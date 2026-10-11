---
title: "Understanding RAGAS Evaluators"
description: "How LLM-based evaluators work in RAG evaluation, and where they break down."
date: "2026-09-24"
updated: "2026-09-25"
category: "AI"
tags: ["RAG", "RAGAS", "LLM", "Evaluation"]
status: "published"
featured: true
readingTime: 8
---

## Introduction

RAGAS scores a RAG pipeline by asking an LLM to judge properties like faithfulness and answer relevancy, rather than comparing against a single fixed reference answer.

## Why RAGAS

Traditional metrics like BLEU or ROUGE assume there's one correct phrasing. RAG answers rarely have one — RAGAS instead measures whether the answer is grounded in the retrieved context and whether it actually addresses the question.

## Evaluation pipeline

A typical run generates answers over a fixed question set, retrieves supporting context, and scores each answer with a panel of judge prompts.

```python
def score(question, answer, context):
    faithfulness = judge_faithfulness(answer, context)
    relevancy = judge_relevancy(answer, question)
    return {"faithfulness": faithfulness, "relevancy": relevancy}
```

## Experiments

Swapping the judge model changes absolute scores but preserves relative ranking between pipeline variants in most of my runs — useful for regression testing even when the judge isn't perfectly calibrated.

## Results

Faithfulness was the most sensitive metric to retrieval quality; answer relevancy tracked prompt phrasing more than retrieval.

## Lessons learned

> Treat LLM judges as a noisy but directionally useful signal — pair them with a small set of hand-labeled examples you trust.

See also [[RAG Evaluation Pipeline Notes]] for the harness this feeds into.
