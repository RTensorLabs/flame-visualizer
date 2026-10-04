/// <reference types="svelte" />
/// <reference types="vite/client" />
interface ImportMetaEnv {
  readonly VITE_MODEL_BASE?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
