/** Promise-based client for the engine worker. */
import type { ModelInfo, Request, Response, TraceManifest } from './types';

type Pending = { resolve: (r: Response) => void; reject: (e: Error) => void; onProgress?: (r: Response) => void };

export class EngineClient {
  private worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  private seq = 0;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker.onmessage = (ev: MessageEvent<{ id: number; msg: Response }>) => {
      const p = this.pending.get(ev.data.id);
      if (!p) return;
      const msg = ev.data.msg;
      if (msg.kind === 'progress') return p.onProgress?.(msg);
      this.pending.delete(ev.data.id);
      if (msg.kind === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg);
    };
    this.worker.onerror = (ev) => {
      for (const p of this.pending.values()) p.reject(new Error(ev.message || 'engine worker crashed'));
      this.pending.clear();
    };
  }

  private call(req: Request, onProgress?: (r: Response) => void): Promise<Response> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, req });
    });
  }

  async load(baseUrl: string, onProgress: (loaded: number, total: number) => void): Promise<ModelInfo> {
    const r = await this.call({ kind: 'load', baseUrl }, (m) => m.kind === 'progress' && onProgress(m.loaded, m.total));
    if (r.kind !== 'loaded') throw new Error('unexpected response');
    return r.info;
  }

  async encode(text: string): Promise<number[]> {
    const r = await this.call({ kind: 'encode', text });
    return r.kind === 'ids' ? r.ids : [];
  }

  async decode(ids: number[]): Promise<string> {
    const r = await this.call({ kind: 'decode', ids });
    return r.kind === 'text' ? r.text : '';
  }

  async logits(ids: number[]): Promise<Float32Array> {
    const r = await this.call({ kind: 'logits', ids });
    if (r.kind !== 'logits') throw new Error('unexpected response');
    return r.logits;
  }

  async trace(ids: number[], lensTopK = 5): Promise<{ manifest: TraceManifest; data: Float32Array; ms: number }> {
    const r = await this.call({ kind: 'trace', ids, lensTopK });
    if (r.kind !== 'trace') throw new Error('unexpected response');
    return r;
  }
}
