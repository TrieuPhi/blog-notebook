---
title: "Learning Python Deeply — Study Notes"
description: "Study notes on descriptors, metaclasses, and the parts of Python I kept skipping."
date: "2026-08-14"
updated: "2026-08-14"
category: "Learning"
tags: ["Learning", "Python"]
status: "note"
featured: false
readingTime: 7
---

## Descriptors

A descriptor is any object implementing `__get__`, `__set__`, or `__delete__` — properties are the common case, but the protocol goes further.
