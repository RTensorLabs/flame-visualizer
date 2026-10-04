/** Typed, zero-copy views over the engine's activation arena. */
import type { TraceManifest } from './engine/types';

export class TensorView {
  readonly strides: number[];
  constructor(
    readonly shape: number[],
    readonly data: Float32Array,
  ) {
    this.strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  }
  at(...idx: number[]): number {
    let o = 0;
    for (let k = 0; k < idx.length; k++) o += idx[k] * this.strides[k];
    return this.data[o];
  }
  /** Contiguous sub-array for the given leading indices, e.g. row(i) of a (T, D) tensor. */
  sub(...idx: number[]): Float32Array {
    let o = 0;
    for (let k = 0; k < idx.length; k++) o += idx[k] * this.strides[k];
    const n = idx.length ? this.strides[idx.length - 1] : this.data.length;
    return this.data.subarray(o, o + n);
  }
}

export class Trace {
  readonly ids: number[];
  readonly T: number;
  private views = new Map<string, TensorView>();
  constructor(
    manifest: TraceManifest,
    readonly data: Float32Array,
    readonly ms: number,
  ) {
    this.ids = manifest.ids;
    this.T = manifest.ids.length;
    for (const e of manifest.entries) {
      const n = e.shape.reduce((a, b) => a * b, 1);
      this.views.set(e.name, new TensorView(e.shape, data.subarray(e.offset, e.offset + n)));
    }
  }
  t(name: string): TensorView {
    const v = this.views.get(name);
    if (!v) throw new Error(`trace has no tensor ${name}`);
    return v;
  }
  L(layer: number, name: string): TensorView {
    return this.t(`L${layer}.${name}`);
  }
  has(name: string): boolean {
    return this.views.has(name);
  }
}
