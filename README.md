# FLAME Visualizer

- **Live:** https://huggingface.co/spaces/Tejascodex001/flame-visualizer
- **Model:** https://huggingface.co/Tejascodex001/flame-tinystories-5m

An interactive look inside FLAME's language model. It runs the real trained model **in the browser** (Rust →
WebAssembly) and visualises every value of the forward pass: token embeddings, Q/K/V, attention patterns, SwiGLU neurons, the
residual stream (logit lens) and the next-token distribution.

## Layout

| path | what |
|---|---|
| `engine/` | `flame-engine`: dependency-light Rust inference + activation tracing (safetensors loader, FLAME-compatible BPE tokenizer, LLaMA-style decoder). Compiles to native and `wasm32`. |
| `parity/` | Own workspace. Tests `flame-engine` against FLAME (path dependency, read-only) on the real checkpoint. |
| `web/` | Svelte 5 + TypeScript + Vite app. The engine runs in a Web Worker; canvas for dense grids, SVG for the diagram. |
| `tools/export_hf.py` | Packages a FLAME run for the Hub (`model.safetensors`, `config.json`, `flame_tokenizer.json`, `metrics.jsonl`, model card). |
| `tools/wasm_smoke.mjs` | Loads the wasm engine in Node and checks greedy generation against FLAME's output. |
| `tools/deploy.sh` | Parity → export → upload model → build pinned to the model commit → upload Space → e2e against the live site. |
| `hf/` | Model card and Space README templates. |

## Guarantees (all automated)

- **Tokenizer:** identical ids to FLAME on 2,427 texts / 503k tokens of TinyStories, plus edge cases (unicode, digits, whitespace); identical decoding of all 4,096 ids.
- **Logits:** max |Δ| = 1.3e-5 vs FLAME at every position of 5 prompts.
- **Generation:** 40 greedy tokens identical to `flame::generate` (native and in WebAssembly).
- **Trace consistency:** attention rows sum to 1 and are 0 in the future; Σ per-head contributions = attention output; block outputs chain; causality.
- **UI:** `svelte-check`, ESLint, Prettier, unit tests, and Playwright e2e with **axe-core WCAG 2.1 AA audits of every view in light and dark themes** (zero violations). Lighthouse on the live site: Performance 96, Accessibility 100, Best Practices 100, SEO 100.

## Develop

```bash
# once
rustup target add wasm32-unknown-unknown && cargo install wasm-pack
python3 tools/export_hf.py --run ../FLAME-main/FLAME/runs/tinystories-5m \
  --tokenizer ../FLAME-main/FLAME/data/tokenizers/tinystories-4k.json --out hf_export --repo Tejascodex001/flame-tinystories-5m
cd web && npm ci && npm run wasm

npm run dev          # http://localhost:5173, model served from ../hf_export at /model/
npm run check && npm run lint && npm test && npm run test:e2e
cargo test -p flame-engine                                   # engine unit tests
cargo test --manifest-path ../parity/Cargo.toml --release    # parity vs FLAME (needs the FLAME checkout + run)
```

## Deploy

```bash
uvx --from huggingface_hub hf auth login     # token with write access
FLAME_DIR=../FLAME-main/FLAME HF_USER={HF_USER} tools/deploy.sh
```

The site is a static bundle (~0.8 MB). The 23 MB model is served by the Hugging Face CDN, pinned to a commit
(`web/.env.production`), and cached in the browser after the first visit. The same `web/dist` folder can be hosted on any static
host (Cloudflare Pages, Vercel, GitHub Pages) without changes.

### Publish website-only changes

The `ci` workflow can also be run manually from `main`. It builds and tests the
existing pinned model and website, then publishes only the static Space files.
It does not export or upload a model. Add `hugging_face` as a repository Actions
secret with write access to `Tejascodex001/flame-visualizer`, then use **Actions →
ci → Run workflow → main**. Ordinary pushes and pull requests run checks without
receiving the Hugging Face credential or publishing anything.

The shared logo marks are in `web/public/brand/`; the Space header, footer,
favicon, and Space README use those same files.
