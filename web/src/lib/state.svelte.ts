/** Global application state (Svelte 5 runes) and the actions that change it. */
import { EngineClient } from './engine/client';
import type { ModelInfo } from './engine/types';
import { PALETTES, type Theme } from './colors';
import { Trace } from './trace';
import { distribution, rng, sample, type SampleSettings } from './sampling';

export type View = 'overview' | 'learn' | 'explore';
export type Stage = 'tokens' | 'embedding' | 'block' | 'output';
export type BlockTab = 'attention' | 'mlp' | 'stream';

export interface Inspect {
  tag: string;
  title: string;
  body: string;
  calc?: string;
}

/** Visualisations show at most this many tokens (the model itself supports 256). */
export const MAX_TOKENS = 64;
export const PRESETS = [
  'Once upon a time, there was a little girl named Lily. She loved to',
  'One day, a boy named Tom went to the park. He saw a big',
  'The cat sat on the mat. The dog',
  'Sue had a red ball. She wanted to',
];

const MODEL_BASE: string = import.meta.env.VITE_MODEL_BASE ?? '/model/';

class App {
  view = $state<View>('overview');
  theme = $state<Theme>('light');
  status = $state<'loading' | 'ready' | 'error'>('loading');
  progress = $state(0);
  error = $state('');
  info = $state.raw<ModelInfo | null>(null);

  text = $state(PRESETS[0]);
  ids = $state.raw<number[]>([]);
  truncated = $state(0);
  trace = $state.raw<Trace | null>(null);
  busy = $state(false);
  lastAdded = $state(-1);

  focus = $state(0);
  step = $state(0);
  stage = $state<Stage>('block');
  block = $state(5);
  head = $state(0);
  tab = $state<BlockTab>('attention');
  zoom = $state<{ i: number; j: number } | null>(null);

  sampling = $state<SampleSettings>({ temperature: 0.8, topK: 40, topP: 1 });
  seed = $state(42);
  generated = 0;

  inspect = $state<Inspect | null>(null);

  get palette() {
    return PALETTES[this.theme];
  }
  get cfg() {
    return this.info?.config ?? null;
  }
  /** Display form with visible spaces (for charts and token chips). */
  tok(id: number): string {
    return this.info?.tokenDisplay[id] ?? String(id);
  }
  /** Plain form for running text: surrounding whitespace trimmed. */
  word(id: number): string {
    const t = this.info?.tokenText[id]?.trim();
    return t || this.tok(id);
  }
}

export const app = new App();
const engine = new EngineClient();
let traceSeq = 0;
let encodeTimer: ReturnType<typeof setTimeout> | undefined;

export async function boot() {
  const saved = localStorage.getItem('flame-theme') as Theme | null;
  setTheme(saved ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'), false);
  routeFromHash();
  addEventListener('hashchange', routeFromHash);
  await loadModel();
}

export async function loadModel() {
  app.status = 'loading';
  app.error = '';
  try {
    if (typeof WebAssembly !== 'object') throw new Error('This browser does not support WebAssembly.');
    const info = await engine.load(MODEL_BASE, (a, b) => (app.progress = b ? a / b : 0));
    app.info = info;
    app.block = info.config.n_layers - 1;
    app.status = 'ready';
    await setText(app.text, true);
  } catch (e) {
    app.status = 'error';
    app.error = e instanceof Error ? e.message : String(e);
  }
}

function routeFromHash() {
  const v = location.hash.replace(/^#\/?/, '').split('/')[0];
  app.view = v === 'learn' || v === 'explore' ? v : 'overview';
}

export function go(view: View) {
  if (location.hash !== `#/${view}`) location.hash = `#/${view}`;
  app.view = view;
  scrollTo({ top: 0 });
}

export function setTheme(t: Theme, persist = true) {
  app.theme = t;
  document.documentElement.dataset.theme = t;
  if (persist) localStorage.setItem('flame-theme', t);
}

/** Re-encode the prompt (debounced unless `now`) and trace it. */
export function setText(text: string, now = false): Promise<void> {
  app.text = text;
  clearTimeout(encodeTimer);
  return new Promise((resolve) => {
    const run = async () => {
      if (app.status !== 'ready') return resolve();
      let ids = await engine.encode(text);
      if (!ids.length) ids = [256]; // empty prompt starts from <|endoftext|>, like FLAME's generate
      app.truncated = Math.max(0, ids.length - MAX_TOKENS);
      app.lastAdded = -1;
      await setIds(ids.slice(-MAX_TOKENS), true);
      resolve();
    };
    if (now) void run();
    else encodeTimer = setTimeout(run, 250);
  });
}

async function setIds(ids: number[], resetFocus: boolean) {
  app.ids = ids;
  const seq = ++traceSeq;
  app.busy = true;
  try {
    const r = await engine.trace(ids);
    if (seq !== traceSeq) return; // a newer request superseded this one
    app.trace = new Trace(r.manifest, r.data, r.ms);
    if (resetFocus || app.focus >= ids.length) app.focus = ids.length - 1;
    if (!app.zoom || app.zoom.i >= ids.length) app.zoom = { i: ids.length - 1, j: Math.max(0, ids.length - 2) };
  } catch (e) {
    app.error = e instanceof Error ? e.message : String(e);
  } finally {
    if (seq === traceSeq) app.busy = false;
  }
}

/** Sample `n` tokens with the current settings, then trace the result once. */
export async function generate(n = 1, pick?: number) {
  if (!app.trace || app.busy) return;
  let ids = [...app.ids];
  const base = app.text;
  const added: number[] = [];
  const random = rng(app.seed + app.generated++);
  app.busy = true;
  for (let k = 0; k < n; k++) {
    let next: number;
    if (k === 0 && pick !== undefined) next = pick;
    else {
      const logits = k === 0 ? app.trace.t('logits_last').data : await engine.logits(ids);
      next = sample(distribution(logits, app.sampling), random);
    }
    if (next === 256) break;
    ids.push(next);
    added.push(next);
    if (ids.length > MAX_TOKENS) {
      ids = ids.slice(1);
      app.truncated++;
    }
    app.ids = ids;
    app.text = base + (await engine.decode(added));
  }
  app.lastAdded = ids.length - 1;
  await setIds(ids, true);
}

export async function reset() {
  app.zoom = null;
  await setText(PRESETS[0], true);
}

export function setFocus(i: number) {
  if (!app.trace) return;
  app.focus = Math.max(0, Math.min(app.trace.T - 1, i));
  if (app.zoom) app.zoom = { i: app.focus, j: Math.min(app.zoom.j, app.focus) };
}
