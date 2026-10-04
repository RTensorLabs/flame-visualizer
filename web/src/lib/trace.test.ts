import { describe, expect, it } from 'vitest';
import { Trace } from './trace';

describe('Trace views', () => {
  const data = Float32Array.from({ length: 2 * 3 + 2 * 2 * 2 }, (_, i) => i);
  const t = new Trace(
    {
      ids: [5, 6],
      entries: [
        { name: 'embed', offset: 0, shape: [2, 3] },
        { name: 'L0.probs', offset: 6, shape: [2, 2, 2] },
      ],
    },
    data,
    1,
  );
  it('indexes row-major', () => {
    expect(t.t('embed').at(1, 2)).toBe(5);
    expect(t.L(0, 'probs').at(1, 0, 1)).toBe(6 + 4 + 1);
  });
  it('sub returns zero-copy rows', () => {
    const r = t.t('embed').sub(1);
    expect(Array.from(r)).toEqual([3, 4, 5]);
    expect(r.buffer).toBe(data.buffer);
  });
  it('throws on unknown tensors', () => {
    expect(() => t.t('nope')).toThrow();
  });
});
