// Loads the wasm engine in Node against an exported model folder and checks greedy generation
// against the text produced by the native parity test (FLAME reference).
import { readFileSync } from "node:fs";
import { initSync, Engine } from "../web/src/lib/wasm/pkg/flame_engine.js";

const dir = process.argv[2] ?? "hf_export";
initSync({ module: readFileSync(new URL("../web/src/lib/wasm/pkg/flame_engine_bg.wasm", import.meta.url)) });
const e = new Engine(readFileSync(`${dir}/model.safetensors`), readFileSync(`${dir}/config.json`, "utf8"), readFileSync(`${dir}/flame_tokenizer.json`, "utf8"));
let ids = Array.from(e.encode("Once upon a time"));
const out = [];
for (let i = 0; i < 40; i++) {
  const l = e.logits_last(Uint32Array.from(ids));
  let b = 0;
  for (let j = 1; j < l.length; j++) if (l[j] > l[b]) b = j;
  ids.push(b); out.push(b);
}
const text = e.decode(Uint32Array.from(out));
const expected = ", there was a little girl named Lily. She had a big, red ball. Lily loved to play with her ball every day. One day, she saw a small, helpless bird on the ground";
console.log("greedy:", JSON.stringify(text));
if (text !== expected) { console.error("MISMATCH vs FLAME reference"); process.exit(1); }
const probe = Uint32Array.from(e.encode("Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big dog in the park. The dog was very friendly"));
for (const T of [16, 32, 48]) {
  const t0 = performance.now(); const tr = e.trace(probe.slice(0, T), 5, true); const ms = performance.now() - t0;
  const n = tr.data().length; tr.free();
  console.log(`trace T=${T}: ${ms.toFixed(0)} ms, ${(n * 4 / 1e6).toFixed(1)} MB`);
}
console.log("OK: wasm engine matches FLAME greedy output");
