import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig, type Connect, type Plugin } from 'vite';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

/** Local dev/preview: serve the exported model folder (../hf_export) at /model/ so the app works offline. */
function localModel(): Plugin {
  const dir = resolve(import.meta.dirname, '../hf_export');
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const file = resolve(dir, (req.url ?? '').replace(/^\//, '').split('?')[0]);
    if (!file.startsWith(dir) || !existsSync(file)) return next();
    res.setHeader('Content-Length', statSync(file).size);
    createReadStream(file).pipe(res);
  };
  return {
    name: 'local-model',
    configureServer: (server) => void server.middlewares.use('/model/', handler),
    configurePreviewServer: (server) => void server.middlewares.use('/model/', handler),
  };
}

export default defineConfig({
  base: './',
  plugins: [svelte(), localModel()],
  worker: { format: 'es' },
  build: { target: 'es2022', assetsInlineLimit: 0 },
});
