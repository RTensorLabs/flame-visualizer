---
title: FLAME Visualizer
emoji: 🔥
colorFrom: yellow
colorTo: red
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: See inside a 5.9M-param LLM, every number, in your browser
models:
  - Tejascodex001/flame-tinystories-5m
datasets:
  - roneneldan/TinyStories
tags:
  - visualization
  - interpretability
  - transformer
  - webassembly
  - education
---

# FLAME Visualizer

An interactive look inside [FLAME TinyStories-5M](https://huggingface.co/Tejascodex001/flame-tinystories-5m), a small
LLaMA-style language model trained from scratch with [FLAME](https://github.com/RTensorLabs/FLAME).

The model runs **entirely in your browser** (Rust → WebAssembly); nothing you type is sent anywhere.
Every square, dot and ribbon is a real value from the forward pass: embeddings, Q/K/V, attention patterns,
SwiGLU neurons, the logit lens and the next-token distribution.

- **Overview**: the whole model as one animated diagram, with a guided "play the forward pass" walkthrough.
- **Learn**: four short steps from tokens to a prediction.
- **Explore**: every block, head and neuron, including the exact query·key arithmetic behind any attention weight.

The in-browser engine is an independent re-implementation that is tested against FLAME itself:
identical tokenization on 500k tokens, logits within 1.4e-5, and identical greedy generations.
