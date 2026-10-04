/** Shapes shared between the main thread and the engine worker. */

export interface ModelConfig {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  mlp_hidden: number;
  max_seq_len: number;
  rope_theta: number;
  norm_eps: number;
  tie_embeddings: boolean;
}

export interface ManifestEntry {
  name: string;
  offset: number;
  shape: number[];
}

export interface TraceManifest {
  ids: number[];
  entries: ManifestEntry[];
}

export interface ModelInfo {
  config: ModelConfig;
  /** Display form of every token id (spaces/newlines made visible, invalid UTF-8 as <0xNN>). */
  tokenDisplay: string[];
  /** Raw decoded text of every token id (for building strings). */
  tokenText: string[];
  /** Bytes of every token id, hex encoded, for the token inspector. */
  tokenHex: string[];
  source: string;
}

export type Request =
  | { kind: 'load'; baseUrl: string }
  | { kind: 'encode'; text: string }
  | { kind: 'decode'; ids: number[] }
  | { kind: 'trace'; ids: number[]; lensTopK: number }
  | { kind: 'logits'; ids: number[] };

export type Response =
  | { kind: 'progress'; loaded: number; total: number; file: string }
  | { kind: 'loaded'; info: ModelInfo }
  | { kind: 'ids'; ids: number[] }
  | { kind: 'text'; text: string }
  | { kind: 'trace'; manifest: TraceManifest; data: Float32Array; ms: number }
  | { kind: 'logits'; logits: Float32Array }
  | { kind: 'error'; message: string };
