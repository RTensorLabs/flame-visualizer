import { describe, expect, it } from 'vitest';
import { distribution, rng, sample, softmax } from './sampling';

describe('sampling (mirrors FLAME generate.rs)', () => {
  const logits = [1, 3, 2, -1];
  it('softmax sums to 1 and keeps order', () => {
    const p = softmax(logits);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[1]).toBeGreaterThan(p[2]);
  });
  it('temperature 0 is greedy', () => {
    const d = distribution(logits, { temperature: 0, topK: 0, topP: 1 });
    expect(d.kept).toBe(1);
    expect(sample(d, rng(1))).toBe(1);
  });
  it('top-k keeps exactly k tokens', () => {
    const d = distribution(logits, { temperature: 1, topK: 2, topP: 1 });
    expect(d.kept).toBe(2);
    expect(d.q[3]).toBe(0);
    expect(d.q[1] + d.q[2]).toBeCloseTo(1, 6);
  });
  it('top-p keeps the smallest nucleus reaching p', () => {
    const d = distribution(logits, { temperature: 1, topK: 0, topP: 0.5 });
    expect(d.kept).toBe(1);
  });
  it('seeded rng is deterministic', () => {
    const a = rng(42),
      b = rng(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});
