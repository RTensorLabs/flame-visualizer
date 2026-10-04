<!-- Small, decorative attention-pattern thumbnail (the parent control carries the accessible name). -->
<script lang="ts">
  import { app } from '../state.svelte';
  import { css, sequential } from '../colors';
  let { probs, head, size = 84 }: { probs: { at: (...i: number[]) => number; shape: number[] }; head: number; size?: number } = $props();
  let cv: HTMLCanvasElement | undefined = $state();
  $effect(() => {
    if (!cv) return;
    const T = probs.shape[1];
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = size * dpr;
    cv.height = size * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const c = size / T;
    const p = app.palette;
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        ctx.fillStyle = css(sequential(p, p.attn, Math.pow(probs.at(head, i, j), 0.7)));
        ctx.fillRect(j * c, i * c, Math.max(1, c - (T > 24 ? 0 : 1)), Math.max(1, c - (T > 24 ? 0 : 1)));
      }
  });
</script>

<canvas bind:this={cv} style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas>

<style>
  canvas {
    display: block;
    border-radius: 4px;
  }
</style>
