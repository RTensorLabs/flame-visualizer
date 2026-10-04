---
license: mit
language:
  - en
library_name: flame
pipeline_tag: text-generation
tags:
  - flame
  - rust
  - candle
  - tinystories
  - llama-style
  - tiny-llm
  - educational
datasets:
  - roneneldan/TinyStories
---

# FLAME TinyStories-5M

A {{PARAMS}}-parameter decoder-only language model trained from scratch with
[FLAME](https://github.com/RTensorLabs/FLAME) (Framework for Lightweight AI Model Engineering),
a small Rust + [Candle](https://github.com/huggingface/candle) training framework, on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset of simple children's stories.

**See inside it:** the interactive visualizer runs this exact model in your browser and shows
every intermediate value of the forward pass: {{SPACE_LINK}}

## Architecture

LLaMA-style pre-norm Transformer.

| | |
|---|---|
| Parameters | {{PARAMS}} (tied input/output embeddings) |
| Layers | {{L}} |
| d_model | {{D}} |
| Heads | {{H}} × {{HD}} dims |
| MLP | SwiGLU, hidden {{F}} |
| Norm | RMSNorm (ε = 1e-5), pre-norm, final norm |
| Positions | RoPE (θ = 10000, rotate-half) |
| Vocabulary | {{V}} (byte-level BPE trained on TinyStories) |
| Context | {{CTX}} tokens |

```
x = tok_emb[ids]
for each block:  x += Wo·Attn(RoPE(Wq·n), RoPE(Wk·n), Wv·n)   n = RMSNorm(x)
                 x += Wdown·(silu(Wgate·m) ⊙ Wup·m)          m = RMSNorm(x)
logits = RMSNorm(x) · tok_embᵀ
```

## Training

| | |
|---|---|
| Steps | {{STEPS}} |
| Tokens seen | {{TOKENS}} |
| Batch | {{BATCH}} sequences × {{SEQ}} tokens |
| Optimiser | AdamW, lr {{LR}} → {{MIN_LR}} (cosine), {{WARMUP}} warmup steps |
| Hardware / time | single laptop GPU (RTX 3070 Ti), {{HOURS}} h |
| Final train loss | {{TRAIN_LOSS}} |
| Best validation loss | {{VAL_LOSS}} (step {{VAL_STEP}}) |

Full training log: `metrics.jsonl`.

## Files

| file | contents |
|---|---|
| `model.safetensors` | F32 weights, FLAME tensor names (`tok_emb.weight`, `layers.{i}.attn.q_proj.weight`, …). SHA-256 `{{SHA256}}` |
| `config.json` | architecture hyper-parameters |
| `flame_tokenizer.json` | FLAME's BPE tokenizer (`{"kind":"bpe","merges":[...]}`); **not** the HF `tokenizers` format |
| `metrics.jsonl` | training / validation log |

Token ids: 0–255 raw bytes, 256 `<|endoftext|>`, 257 `<|pad|>`, 258+ BPE merges.

## Usage

With FLAME: `model.safetensors` is the unmodified FLAME checkpoint (`runs/tinystories-5m/best/model.safetensors`).
Inside a FLAME run directory (`config.toml` + `best/model.safetensors`, tokenizer path from `[data]`):

```bash
flame generate --run runs/tinystories-5m --prompt "Once upon a time"
```

In the browser (WebAssembly) with the standalone engine from the visualizer project:

```js
import init, { Engine } from "./flame_engine.js";
await init();
const base = "https://huggingface.co/{{REPO}}/resolve/main/";
const [w, c, t] = await Promise.all(["model.safetensors", "config.json", "flame_tokenizer.json"]
  .map((f, i) => fetch(base + f).then(r => i ? r.text() : r.arrayBuffer())));
const engine = new Engine(new Uint8Array(w), c, t);
const logits = engine.logits_last(engine.encode("Once upon a time"));
```

## Limitations

- Trained only on synthetic children's stories: simple English vocabulary, short narratives.
  It is not a general-purpose or instruction-following model and knows little about the world.
- 256-token context.
- Outputs can be repetitive or incoherent, and may reflect biases of the generated training data.
- Intended for education and research on small language models.

## License

Weights and code: MIT. Training data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), CDLA-Sharing-1.0 (which does not restrict results produced by computing on the data, such as trained weights).
