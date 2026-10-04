/**
 * Sampling, mirroring FLAME's generate.rs: temperature ≤ 0 is greedy (argmax); top-k keeps the k
 * highest logits (0 = all); probabilities are softmax((logit − max) / T).
 * Top-p (nucleus) is NOT part of FLAME; it is offered as an extra visualiser filter.
 */
export interface SampleSettings {
  temperature: number;
  topK: number;
  topP: number;
}

export interface Distribution {
  /** Token ids sorted by probability (highest first). */
  order: number[];
  /** softmax(logits / T) over the whole vocabulary (T clamped to a tiny value when greedy). */
  p: Float32Array;
  /** Renormalised probability after top-k / top-p; 0 for filtered tokens. */
  q: Float32Array;
  kept: number;
}

export function softmax(logits: ArrayLike<number>, temperature = 1): Float32Array {
  const t = Math.max(temperature, 1e-4);
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) m = Math.max(m, logits[i]);
  const out = new Float32Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) s += out[i] = Math.exp((logits[i] - m) / t);
  for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}

export function distribution(logits: ArrayLike<number>, s: SampleSettings): Distribution {
  const p = softmax(logits, s.temperature);
  const order = Array.from({ length: p.length }, (_, i) => i).sort((a, b) => p[b] - p[a]);
  const q = new Float32Array(p.length);
  if (s.temperature <= 0) {
    q[order[0]] = 1;
    return { order, p, q, kept: 1 };
  }
  const k = s.topK > 0 ? Math.min(s.topK, p.length) : p.length;
  let cum = 0;
  let kept = 0;
  for (let r = 0; r < k; r++) {
    const id = order[r];
    if (r > 0 && cum >= s.topP) break;
    q[id] = p[id];
    cum += p[id];
    kept++;
  }
  for (let i = 0; i < q.length; i++) q[i] /= cum;
  return { order, p, q, kept };
}

/** Deterministic PRNG (mulberry32) so a seed reproduces a generation. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sample(d: Distribution, random: () => number): number {
  let r = random();
  for (const id of d.order) {
    if (d.q[id] === 0) continue;
    r -= d.q[id];
    if (r <= 0) return id;
  }
  return d.order[0];
}
