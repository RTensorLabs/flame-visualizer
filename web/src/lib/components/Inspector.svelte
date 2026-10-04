<script lang="ts">
  import { app } from '../state.svelte';
</script>

<section class="card insp" aria-labelledby="insp-h">
  <h2 id="insp-h" class="eyebrow">{app.inspect?.tag ?? 'Inspector'}</h2>
  <div aria-live="polite">
    {#if app.inspect}
      <p class="t">{app.inspect.title}</p>
      <!-- eslint-disable-next-line svelte/no-at-html-tags -- trusted, generated strings -->
      <p class="b">{@html app.inspect.body}</p>
      {#if app.inspect.calc}<p class="calc">{app.inspect.calc}</p>{/if}
    {:else}
      <p class="t">Hover or focus anything coloured</p>
      <p class="b">Every square is a real number from this forward pass. Use the mouse, or Tab to a chart and use the arrow keys.</p>
      <ul class="legend">
        <li><i style:background="var(--pos)"></i>positive</li>
        <li><i style:background="var(--neg)"></i>negative</li>
        <li><i style:background="var(--cell)" style:outline="1px solid var(--border2)"></i>≈ 0</li>
      </ul>
    {/if}
  </div>
</section>

<style>
  .insp {
    padding: 16px 18px;
    min-height: 190px;
  }
  .t {
    margin: 8px 0 6px;
    font-weight: 650;
    font-size: 14.5px;
    line-height: 1.3;
  }
  .b {
    margin: 0;
    font-size: 13.5px;
    color: var(--muted);
  }
  .calc {
    margin: 12px 0 0;
    font: 500 12.5px var(--mono);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 9px 12px;
    overflow-x: auto;
    white-space: nowrap;
  }
  .legend {
    display: flex;
    gap: 14px;
    margin: 14px 0 0;
    padding: 0;
    list-style: none;
    font-size: 12.5px;
    color: var(--muted);
  }
  .legend i {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    margin-right: 6px;
    vertical-align: -1px;
  }
</style>
