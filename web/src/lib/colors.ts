/**
 * Colour scales shared by canvas and SVG. Values are mixed from the theme's neutral "cell"
 * colour towards a hue, so zero is always neutral and both themes stay legible.
 */
export type RGB = [number, number, number];
export type Theme = 'light' | 'dark';

export interface Palette {
  cell: RGB;
  pos: RGB;
  neg: RGB;
  attn: RGB;
  out: RGB;
  ffn: RGB;
  emb: RGB;
  q: RGB;
  k: RGB;
  v: RGB;
  mix: RGB;
  text: RGB;
}

export const PALETTES: Record<Theme, Palette> = {
  light: {
    cell: [236, 235, 231],
    pos: [234, 88, 12],
    neg: [37, 99, 235],
    attn: [234, 88, 12],
    out: [37, 99, 235],
    ffn: [13, 148, 136],
    emb: [124, 58, 237],
    q: [37, 99, 235],
    k: [220, 38, 38],
    v: [5, 150, 105],
    mix: [147, 51, 234],
    text: [24, 24, 27],
  },
  dark: {
    cell: [36, 36, 42],
    pos: [251, 122, 60],
    neg: [96, 146, 255],
    attn: [251, 122, 60],
    out: [96, 146, 255],
    ffn: [45, 212, 191],
    emb: [167, 139, 250],
    q: [96, 146, 255],
    k: [248, 113, 113],
    v: [52, 211, 153],
    mix: [192, 132, 252],
    text: [237, 237, 240],
  },
};

const lerp = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const css = (c: RGB, alpha = 1) =>
  alpha === 1 ? `rgb(${c[0] | 0} ${c[1] | 0} ${c[2] | 0})` : `rgb(${c[0] | 0} ${c[1] | 0} ${c[2] | 0} / ${alpha})`;

/** Diverging scale: negative → neg hue, 0 → neutral, positive → pos hue. `scale` maps to full intensity. */
export function diverging(p: Palette, v: number, scale: number): RGB {
  if (!Number.isFinite(v)) return p.cell;
  const t = Math.min(Math.abs(v) / (scale || 1), 1);
  return lerp(p.cell, v >= 0 ? p.pos : p.neg, Math.pow(t, 0.75) * 0.95);
}

/** Sequential scale from neutral to `hue` for values in [0, 1]. */
export function sequential(p: Palette, hue: RGB, a: number): RGB {
  return lerp(p.cell, hue, Math.min(Math.max(a, 0), 1));
}

/** WCAG relative luminance of an sRGB colour. */
export function luminance(c: RGB): number {
  const f = (v: number) => {
    const x = v / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
}

/** Text colour (near-black or white) with the higher WCAG contrast on this background. */
export function onColor(c: RGB): string {
  const l = luminance(c);
  const black = (l + 0.05) / (luminance([24, 24, 27]) + 0.05);
  const white = 1.05 / (l + 0.05);
  return black >= white ? '#18181b' : '#ffffff';
}

/** Robust colour scale for a set of values: the 98th percentile of |v| (ignores NaN). */
export function robustScale(values: ArrayLike<number>): number {
  const a: number[] = [];
  const step = Math.max(1, Math.floor(values.length / 4000));
  for (let i = 0; i < values.length; i += step) if (Number.isFinite(values[i])) a.push(Math.abs(values[i]));
  if (!a.length) return 1;
  a.sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.floor(a.length * 0.98))] || 1;
}
