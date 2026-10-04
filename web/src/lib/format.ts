export const pct = (x: number) => (x * 100).toFixed(x < 0.001 ? 2 : x < 0.1 ? 1 : 0) + '%';
export const signed = (x: number, d = 3) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
export const num = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '—');
export const compact = (n: number) => (n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(n));
