/// <reference lib="webworker" />
/**
 * Engine worker: downloads the model once (Cache Storage), runs the WebAssembly engine and
 * answers requests from the main thread. Keeps all heavy work off the UI thread.
 */
import init, { Engine } from '../wasm/pkg/flame_engine.js';
import wasmUrl from '../wasm/pkg/flame_engine_bg.wasm?url';
import type { ModelConfig, Request, Response } from './types';

declare const self: DedicatedWorkerGlobalScope;

const CACHE = 'flame-model-v1';
let engine: Engine | null = null;

function post(id: number, msg: Response, transfer: Transferable[] = []) {
  self.postMessage({ id, msg }, transfer);
}

async function fetchCached(url: string, onBytes: (n: number, total: number) => void): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open(CACHE) : null;
  const hit = await cache?.match(url);
  const res = hit ?? (await fetch(url));
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.clone().body?.getReader();
  if (!reader) return new Uint8Array(await res.arrayBuffer());
  const chunks: Uint8Array[] = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onBytes(n, total);
  }
  if (!hit && cache) await cache.put(url, res).catch(() => undefined);
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function display(bytes: Uint8Array): string {
  try {
    const s = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    return s.replace(/ /g, '·').replace(/\n/g, '⏎').replace(/\t/g, '⇥').replace(/\r/g, '␍');
  } catch {
    return Array.from(bytes, (b) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`).join('');
  }
}

async function load(id: number, baseUrl: string) {
  const files = ['config.json', 'flame_tokenizer.json', 'model.safetensors'];
  const sizes = new Map<string, [number, number]>();
  const report = (file: string) => (n: number, total: number) => {
    sizes.set(file, [n, total]);
    let a = 0;
    let b = 0;
    for (const [x, y] of sizes.values()) {
      a += x;
      b += y || x;
    }
    post(id, { kind: 'progress', loaded: a, total: b, file });
  };
  const [cfgBytes, tokBytes, weights] = await Promise.all(files.map((f) => fetchCached(baseUrl + f, report(f))));
  await init({ module_or_path: wasmUrl });
  const dec = new TextDecoder();
  const cfgText = dec.decode(cfgBytes);
  engine = new Engine(weights, cfgText, dec.decode(tokBytes));
  const config = JSON.parse(engine.config_json()) as ModelConfig;
  config.mlp_hidden = engine.hidden();
  const tokenDisplay: string[] = [];
  const tokenText: string[] = [];
  const tokenHex: string[] = [];
  for (let i = 0; i < config.vocab_size; i++) {
    const b = engine.token_bytes(i);
    tokenDisplay.push(i === 256 ? '<|endoftext|>' : i === 257 ? '<|pad|>' : display(b));
    tokenText.push(new TextDecoder().decode(b));
    tokenHex.push(Array.from(b, (x) => x.toString(16).padStart(2, '0')).join(' '));
  }
  post(id, { kind: 'loaded', info: { config, tokenDisplay, tokenText, tokenHex, source: baseUrl } });
}

self.onmessage = async (ev: MessageEvent<{ id: number; req: Request }>) => {
  const { id, req } = ev.data;
  try {
    if (req.kind === 'load') return await load(id, req.baseUrl);
    if (!engine) throw new Error('model not loaded');
    switch (req.kind) {
      case 'encode':
        return post(id, { kind: 'ids', ids: Array.from(engine.encode(req.text)) });
      case 'decode':
        return post(id, { kind: 'text', text: engine.decode(Uint32Array.from(req.ids)) });
      case 'logits': {
        const logits = engine.logits_last(Uint32Array.from(req.ids));
        return post(id, { kind: 'logits', logits }, [logits.buffer]);
      }
      case 'trace': {
        const t0 = performance.now();
        const r = engine.trace(Uint32Array.from(req.ids), req.lensTopK, true);
        const data = r.data();
        const manifest = JSON.parse(r.manifest());
        r.free();
        return post(id, { kind: 'trace', manifest, data, ms: performance.now() - t0 }, [data.buffer]);
      }
    }
  } catch (e) {
    post(id, { kind: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
