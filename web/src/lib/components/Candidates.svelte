<!-- Next-token candidates from the real 4096-way distribution, with sampling controls. -->
<script lang="ts">
  import { app, generate } from '../state.svelte';
  import { distribution } from '../sampling';
  import { pct } from '../format';
  import { showInspect, hideTip } from '../inspect.svelte';

  interface Props {
    limit?: number;
    controls?: 'none' | 'temperature' | 'all';
    actions?: boolean;
  }
  let { limit = 10, controls = 'all', actions = true }: Props = $props();

  const logits = $derived(app.trace?.t('logits_last').data);
  const dist = $derived(logits ? distribution(logits, app.sampling) : null);
  const raw = $derived(logits ? distribution(logits, { temperature: 1, topK: 0, topP: 1 }) : null);
  const rows = $derived(dist ? dist.order.slice(0, limit) : []);
  const max = $derived(dist ? dist.p[dist.order[0]] : 1);
  const note = $derived(
    app.sampling.temperature <= 0
      ? 'Greedy: always the single most likely token.'
      : app.sampling.temperature < 0.6
        ? 'Cautious: the favourite wins almost every time.'
        : app.sampling.temperature > 1.3
          ? 'Adventurous: unlikely tokens get a real chance.'
          : 'Balanced: likely tokens win, with some variety.',
  );

  function hover(ev: MouseEvent, id: number) {
    if (!dist || !raw || !logits) return;
    showInspect(
      {
        tag: 'Next-token probability',
        title: `“${app.tok(id)}” (token #${id})`,
        body:
          dist.q[id] > 0
            ? `Survives the filters with a <b>${pct(dist.q[id])}</b> chance of being picked.`
            : 'Removed by top-k / top-p, so it cannot be picked.',
        calc: `logit ${logits[id].toFixed(3)} → p ${pct(raw.p[id])} at T=1, ${pct(dist.p[id])} at T=${app.sampling.temperature.toFixed(2)}`,
      },
      ev.clientX,
      ev.clientY,
    );
  }
</script>

{#if dist}
  <ol class="cands">
    {#each rows as id, r (id)}
      <li class:cut={dist.q[id] === 0} class:top={r === 0} onmousemove={(e) => hover(e, id)} onmouseleave={hideTip}>
        <span class="w">{app.tok(id)}</span>
        <span class="track" aria-hidden="true"><i style:width="{(dist.p[id] / max) * 100}%"></i></span>
        <span class="p">{dist.q[id] === 0 ? 'cut' : pct(dist.q[id])}</span>
      </li>
    {/each}
  </ol>
  {#if controls !== 'none'}
    <div class="sliders">
      <label for="temp">Temperature</label>
      <input id="temp" type="range" min="0" max="2" step="0.05" bind:value={app.sampling.temperature} aria-describedby="temp-note" />
      <output for="temp">{app.sampling.temperature.toFixed(2)}</output>
      {#if controls === 'all'}
        <label for="topk">Top-k</label>
        <input id="topk" type="range" min="0" max="100" step="1" bind:value={app.sampling.topK} />
        <output for="topk">{app.sampling.topK || 'off'}</output>
        <label for="topp">Top-p <abbr title="Not part of FLAME's sampler; a visualiser-only extra">*</abbr></label>
        <input id="topp" type="range" min="0.05" max="1" step="0.05" bind:value={app.sampling.topP} />
        <output for="topp">{app.sampling.topP >= 1 ? 'off' : app.sampling.topP.toFixed(2)}</output>
      {/if}
      <p class="note" id="temp-note">{note} Shares shown are after filtering ({dist.kept} token{dist.kept === 1 ? '' : 's'} kept).</p>
    </div>
  {/if}
  {#if actions}
    <div class="acts">
      <button class="btn accent" type="button" disabled={app.busy} onclick={() => generate(1)}>Sample next token</button>
      <button class="btn ghost" type="button" disabled={app.busy} onclick={() => generate(1, dist.order[0])}
        >Add “{app.tok(dist.order[0])}”</button
      >
    </div>
  {/if}
{/if}

<style>
  .cands {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  li {
    display: grid;
    grid-template-columns: minmax(70px, 120px) 1fr 54px;
    gap: 12px;
    align-items: center;
    padding: 5px 8px;
    border-radius: 8px;
  }
  li:hover {
    background: var(--surface2);
  }
  .w {
    font: 600 13.5px var(--mono);
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track {
    height: 12px;
    border-radius: 5px;
  }
  .track > i {
    background: color-mix(in oklab, var(--out) 50%, var(--cell));
    border-radius: 5px;
  }
  .top .track > i {
    background: var(--out);
  }
  .top .w {
    color: var(--out);
  }
  .p {
    font: 500 12px var(--mono);
    text-align: right;
    color: var(--muted);
  }
  .cut {
    opacity: 0.45;
  }
  .cut .w {
    text-decoration: line-through;
  }
  .sliders {
    display: grid;
    grid-template-columns: 100px 1fr 44px;
    gap: 10px;
    align-items: center;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 13.5px;
  }
  .sliders label {
    font-weight: 600;
  }
  .sliders input {
    width: 100%;
    accent-color: var(--out);
  }
  .sliders output {
    font: 500 12px var(--mono);
    text-align: right;
  }
  .sliders .note {
    grid-column: 1 / -1;
    margin: 0;
  }
  .acts {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
  }
</style>
