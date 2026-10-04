/** Small analyses computed from a trace, for labels and summaries. */
import type { Trace } from './trace';

/** Describes what a head mostly attends to, from its (T, T) pattern, averaged over positions ≥ 2. */
export function headHabit(trace: Trace, layer: number, head: number): string {
  const T = trace.T;
  if (T < 3) return 'needs more tokens';
  const P = trace.L(layer, 'probs');
  let prev = 0,
    first = 0,
    self = 0,
    two = 0,
    ent = 0;
  for (let i = 2; i < T; i++) {
    prev += P.at(head, i, i - 1);
    first += P.at(head, i, 0);
    self += P.at(head, i, i);
    two += P.at(head, i, i - 2);
    let h = 0;
    for (let j = 0; j <= i; j++) {
      const p = P.at(head, i, j);
      if (p > 0) h -= p * Math.log(p);
    }
    ent += h / Math.log(i + 1);
  }
  const n = T - 2;
  const scores: [string, number][] = [
    ['the previous token', prev / n],
    ['the first token', first / n],
    ['itself', self / n],
    ['two tokens back', two / n],
  ];
  scores.sort((a, b) => b[1] - a[1]);
  if (scores[0][1] < 0.3) return ent / n > 0.6 ? 'spreads attention widely' : 'mixes a few tokens';
  return `tends to look at ${scores[0][0]}`;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na * nb) || 1);
}

export function norm(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

/** Indices of the k largest |values|. */
export function topAbs(a: Float32Array, k: number): number[] {
  return Array.from(a.keys())
    .sort((x, y) => Math.abs(a[y]) - Math.abs(a[x]))
    .slice(0, k);
}
