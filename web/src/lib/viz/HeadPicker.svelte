<script lang="ts">
  import { app } from '../state.svelte';
  import { headHabit } from '../derived';
  import MiniMatrix from './MiniMatrix.svelte';
  let { layer, value, onpick }: { layer: number; value: number; onpick: (h: number) => void } = $props();
  const probs = $derived(app.trace!.L(layer, 'probs'));
</script>

<div class="heads" role="group" aria-label="Attention heads in block {layer + 1}">
  {#each Array.from({ length: app.cfg!.n_heads }, (_, h) => h) as h (h)}
    <button type="button" class="hc" aria-pressed={h === value} onclick={() => onpick(h)}>
      <b>Head {h + 1}</b>
      <MiniMatrix {probs} head={h} />
      <small>{headHabit(app.trace!, layer, h)}</small>
    </button>
  {/each}
</div>

<style>
  .heads {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(112px, 1fr));
    gap: 8px;
  }
  .hc {
    border: 1px solid var(--border);
    background: var(--surface2);
    border-radius: 10px;
    padding: 9px;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .hc:hover {
    border-color: var(--border2);
  }
  .hc[aria-pressed='true'] {
    border-color: var(--attn);
    background: var(--attn-soft);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--attn) 14%, transparent);
  }
  b {
    font-size: 12.5px;
  }
  small {
    font-size: 11.5px;
    color: var(--muted);
    line-height: 1.3;
    min-height: 30px;
  }
</style>
