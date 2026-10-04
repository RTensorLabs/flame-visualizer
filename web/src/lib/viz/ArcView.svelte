<!-- Attention of one query token over the tokens before it, drawn as arcs (thicker = more). -->
<script lang="ts">
  import { app } from '../state.svelte';
  import { pct } from '../format';
  import { css } from '../colors';

  interface Props {
    weights: Float32Array; // attention row of the query (length T)
    query: number;
    window?: number;
  }
  let { weights, query, window = 14 }: Props = $props();

  const start = $derived(Math.max(0, query - window + 1));
  const hidden = $derived(weights.subarray(0, start).reduce((a, b) => a + b, 0));
  const n = $derived(query - start + 1);
  const W = $derived(Math.max(560, n * 70));
  const x = (j: number) => (j - start + 0.5) * (W / n);
  const base = 170;
  const label = (j: number) => {
    const s = app.trace ? app.tok(app.trace.ids[j]) : '';
    return s.length > 9 ? s.slice(0, 8) + '…' : s;
  };
  const top = $derived(
    Array.from({ length: n }, (_, k) => start + k)
      .sort((a, b) => weights[b] - weights[a])
      .slice(0, 3),
  );
</script>

<figure>
  <svg
    viewBox="0 0 {W} {base + 8}"
    role="img"
    aria-label="Attention from token {query + 1}. Strongest: {top.map((j) => `${label(j)} ${pct(weights[j])}`).join(', ')}."
  >
    {#each Array.from({ length: n }, (_, k) => start + k) as j (j)}
      {@const a = weights[j]}
      {#if a >= 0.01}
        {#if j === query}
          <path
            d="M{x(j) - 9},{base - 20} a 10 15 0 1 1 18 0"
            stroke-width={1 + a * 14}
            opacity={0.25 + a * 0.75}
            stroke={css(app.palette.attn)}
          />
        {:else}
          {@const h = Math.min(140, 28 + Math.abs(x(query) - x(j)) * 0.38)}
          <path
            d="M{x(query)},{base - 20} C{x(query)},{base - 20 - h} {x(j)},{base - 20 - h} {x(j)},{base - 20}"
            stroke-width={1 + a * 14}
            opacity={0.25 + a * 0.75}
            stroke={css(app.palette.attn)}
          />
          {#if a >= 0.06}<text class="p" x={(x(query) + x(j)) / 2} y={base - 30 - h * 0.75} text-anchor="middle">{pct(a)}</text>{/if}
        {/if}
      {/if}
    {/each}
    {#each Array.from({ length: n }, (_, k) => start + k) as j (j)}
      <text class="w" class:on={j === query} x={x(j)} y={base} text-anchor="middle">{label(j)}</text>
    {/each}
  </svg>
  {#if start > 0}<figcaption class="note">+ {start} earlier tokens not drawn ({pct(hidden)} of the attention).</figcaption>{/if}
</figure>

<style>
  figure {
    margin: 0;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
    overflow: visible;
  }
  path {
    fill: none;
    stroke-linecap: round;
  }
  .w {
    font: 500 13px var(--mono);
    fill: var(--muted);
  }
  .w.on {
    fill: var(--accent);
    font-weight: 700;
  }
  .p {
    font: 500 11px var(--mono);
    fill: var(--faint);
  }
</style>
